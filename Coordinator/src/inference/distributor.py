"""
Task distribution — splits feature maps and dispatches work to MCU workers.

Every public ``distribute_*`` method takes the current feature map as input and
returns the resulting output feature map.  The caller (InferenceEngine) never
touches the worker manager directly.
"""

from __future__ import annotations

import asyncio
import logging
import time

import numpy as np

from ..model.types import BlockConfig, LayerConfig, QuantParams
from ..protocol import (
    ErrorMessage,
    LayerType,
    MessageHeader,
    MessageType,
    ResultMessage,
    TaskMessage,
)
from ..stats import InferenceStats
from ..work_manager import WorkerInfo, WorkerManager, WorkerState

logger = logging.getLogger(__name__)


class TaskDistributor:
    """Split data and communicate with workers.

    Stateless with respect to inference progress — the engine owns the feature
    map and residual buffers.  The distributor only borrows the feature map for
    the duration of a single distribute call.
    """

    def __init__(self, worker_manager: WorkerManager, stats: InferenceStats) -> None:
        self.worker_manager = worker_manager
        self.stats = stats

    # ------------------------------------------------------------------
    # Public distribute methods
    # ------------------------------------------------------------------

    async def distribute_conv(self, feature_map: np.ndarray, 
                              layer: LayerConfig, quant_params: QuantParams, current_layer_idx: int,
                              halo_state: tuple | None = None, 
                              return_halo_plan: tuple[list[int], list[int]] | None = None) -> tuple[np.ndarray, list[tuple[int, int] | None] | None]:
        """Split ``feature_map`` by rows and return assembled output.

        When ``halo_state`` is provided, workers may reuse their previously
        cached output rows instead of re-receiving them, mirroring the halo
        reuse path in ``distribute_block``.
        """
        C, H, W = feature_map.shape
        H_out = (H + 2 * layer.padding - layer.kernel_size) // layer.stride + 1
        W_out = (W + 2 * layer.padding - layer.kernel_size) // layer.stride + 1

        available_workers = list(self.worker_manager.workers.values())
        num_workers = len(available_workers)
        rows_per_worker = int(np.ceil(H_out / num_workers))
        tasks: list[tuple[WorkerInfo, int, int, asyncio.Task]] = []
        new_worker_rows: list[tuple[int, int] | None] = [None] * num_workers
        any_skipped = False

        for i, worker in enumerate(available_workers):
            start_row = i * rows_per_worker
            end_row = min(start_row + rows_per_worker, H_out)
            if start_row >= H_out:
                any_skipped = True
                continue

            in_start_y_raw = start_row * layer.stride - layer.padding
            in_end_y_raw = (end_row - 1) * layer.stride + layer.kernel_size - layer.padding

            in_start_y = max(0, in_start_y_raw)
            in_end_y = min(H, in_end_y_raw)

            pad_top = in_start_y - in_start_y_raw
            pad_bottom = in_end_y_raw - in_end_y

            k_top, k_bottom = self._resolve_return_halo(return_halo_plan, i, end_row - start_row)

            halo_usable = (
                halo_state is not None
                and i < len(halo_state[0])
                and halo_state[0][i] is not None
            )
            if halo_usable:
                prev_start, prev_end = halo_state[0][i]
                ov_start = max(in_start_y, prev_start)
                ov_end = min(in_end_y, prev_end)
                if ov_end <= ov_start:
                    halo_usable = False
            if halo_usable:
                logger.debug(
                    f"[Distributor]: Reusing halo cache for worker {worker.worker_id} on layer {current_layer_idx} ({layer.name}), "
                    f"halo rows {ov_start}-{ov_end} (pad_top={pad_top}, pad_bottom={pad_bottom})"
                )
                cache_use_start = ov_start - prev_start
                cache_use_end = ov_end - prev_start
                halo_top = feature_map[:, in_start_y:ov_start, :]
                halo_bottom = feature_map[:, ov_end:in_end_y, :]
                # payload = np.concatenate([halo_top, halo_bottom], axis=1)
                payload_bytes = (np.ascontiguousarray(halo_top).tobytes() + np.ascontiguousarray(halo_bottom).tobytes())
                payload = np.frombuffer(payload_bytes, dtype=np.uint8)
                task_msg = TaskMessage(
                    layer_type=layer.type,
                    layer_idx=current_layer_idx,
                    end_layer_idx=current_layer_idx,
                    in_channels=layer.in_channels,
                    in_h=in_end_y - in_start_y,
                    in_w=W,
                    out_channels=layer.out_channels,
                    out_h=end_row - start_row,
                    out_w=W_out,
                    kernel_size=layer.kernel_size,
                    stride=layer.stride,
                    padding=layer.padding,
                    groups=layer.groups,
                    in_features=0,
                    out_features=0,
                    input_size=payload.size,
                    block_pad_top=pad_top,
                    block_pad_bottom=pad_bottom,
                    use_halo_cache=1,
                    halo_top_rows=halo_top.shape[1],
                    halo_bottom_rows=halo_bottom.shape[1],
                    cache_use_start=cache_use_start,
                    cache_use_end=cache_use_end,
                    prev_block_end_idx=halo_state[1],
                    return_halo=1 if (k_top or k_bottom) else 0,
                    return_halo_top_rows=k_top,
                    return_halo_bottom_rows=k_bottom,
                )
                full_input_bytes = layer.in_channels * (in_end_y - in_start_y) * W
                send_full = full_input_bytes
            else:
                input_patch = feature_map[:, in_start_y:in_end_y, :]
                payload = input_patch
                task_msg = TaskMessage(
                    layer_type=layer.type,
                    layer_idx=current_layer_idx,
                    end_layer_idx=current_layer_idx,
                    in_channels=layer.in_channels,
                    in_h=input_patch.shape[1],
                    in_w=input_patch.shape[2],
                    out_channels=layer.out_channels,
                    out_h=end_row - start_row,
                    out_w=W_out,
                    kernel_size=layer.kernel_size,
                    stride=layer.stride,
                    padding=layer.padding,
                    groups=layer.groups,
                    in_features=0,
                    out_features=0,
                    input_size=input_patch.size,
                    block_pad_top=pad_top,
                    block_pad_bottom=pad_bottom,
                    return_halo=1 if (k_top or k_bottom) else 0,
                    return_halo_top_rows=k_top,
                    return_halo_bottom_rows=k_bottom,
                )
                send_full = None

            task = asyncio.create_task(self._send_task_to_worker(worker, task_msg, payload, current_layer_idx, send_full))
            tasks.append((worker, start_row, end_row, task, k_top, k_bottom))
            new_worker_rows[i] = (start_row, end_row)
            logger.debug(
                f"[Distributor]: Assigned output rows {start_row}-{end_row} to worker {worker.worker_id} for layer {layer.name}"
                f"{f', return halo rows top:{k_top} bottom:{k_bottom}' if (k_top or k_bottom) else ''}"
            )

        await asyncio.gather(*[t[3] for t in tasks])
        output_shape = (layer.out_channels, H_out, W_out)
        output = await self._collect_results(tasks, output_shape, current_layer_idx)
        if any_skipped:
            return output, None
        return output, new_worker_rows


    async def distribute_fc(self, feature_map: np.ndarray, 
                            layer: LayerConfig, quant_params: QuantParams, current_layer_idx: int) -> np.ndarray:
        """Split FC layer by output classes."""
        input_vec = feature_map.flatten()
        total_classes = layer.out_channels
        available_workers = list(self.worker_manager.workers.values())
        num_workers = len(available_workers)
        classes_per_worker = total_classes // num_workers

        logger.debug(
            f"[Distributor]: Distributing FC layer {layer.name} with {total_classes} classes across {num_workers} workers"
        )

        tasks: list[tuple[WorkerInfo, int, int, asyncio.Task]] = []
        for i, worker in enumerate(available_workers):
            start_cls = worker.worker_id * classes_per_worker
            # Last worker gets the remainder so every class is covered
            end_cls = total_classes if worker.worker_id == num_workers - 1 else start_cls + classes_per_worker
            if start_cls >= total_classes:
                continue

            task_msg = TaskMessage(
                layer_type=layer.type,
                layer_idx=current_layer_idx,
                end_layer_idx=current_layer_idx,
                in_channels=layer.in_channels,
                in_h=1,
                in_w=1,
                out_channels=end_cls - start_cls,
                out_h=1,
                out_w=1,
                kernel_size=0,
                stride=0,
                padding=0,
                groups=0,
                in_features=input_vec.size,
                out_features=end_cls - start_cls,
                input_size=input_vec.size,
            )
            task = asyncio.create_task(self._send_task_to_worker(worker, task_msg, input_vec, current_layer_idx))
            # tasks.append((worker, start_cls, end_cls, task))
            tasks.append((worker, start_cls, end_cls, task, 0, 0)) # no return halo
            logger.debug(
                f"[Distributor]: Assigned classes {start_cls}-{end_cls} to worker {worker.worker_id} for FC layer {layer.name}"
            )

        await asyncio.gather(*[t[3] for t in tasks])
        output_shape = (total_classes,)
        return await self._collect_results(tasks, output_shape, current_layer_idx)

    async def distribute_block(self, feature_map: np.ndarray, block: BlockConfig, current_layer_idx: int, 
                               halo_state: tuple | None = None,
                               return_halo_plan: tuple[list[int], list[int]] | None = None) -> tuple[np.ndarray, list[tuple[int, int]] | None]:
        """Distribute a multi-layer block by row-splitting the block input.

        DW padding is applied on the worker side *after* the expand layer.
        """
        C, H, W = feature_map.shape

        # Find the depthwise layer (the only spatial layer in the block)
        dw_layer = None
        for layer in block.layers:
            if layer.type == LayerType.DEPTHWISE:
                dw_layer = layer
                break

        if dw_layer is None:
            # No spatial layer — fall back to running each layer individually
            logger.warning(
                f"[Distributor]: Block [{block.start_idx}-{block.end_idx}] has no DW layer, "
                "falling back to sequential single-layer execution"
            )
            result = feature_map
            for layer, qp in zip(block.layers, block.quant_params):
                if layer.type == LayerType.FC:
                    result = await self.distribute_fc(result, layer, qp, layer.layer_idx)
                else:
                    result, _ = await self.distribute_conv(result, layer, qp, layer.layer_idx)
            return result, None

        dw_padding = dw_layer.padding
        dw_kernel = dw_layer.kernel_size
        dw_stride = dw_layer.stride

        H_out = (H + 2 * dw_padding - dw_kernel) // dw_stride + 1
        W_out = (W + 2 * dw_padding - dw_kernel) // dw_stride + 1
        out_channels = block.layers[-1].out_channels

        available_workers = list(self.worker_manager.workers.values())
        num_workers = len(available_workers)
        rows_per_worker = int(np.ceil(H_out / num_workers))

        tasks: list[tuple[WorkerInfo, int, int, asyncio.Task]] = []
        new_worker_rows: list[tuple[int, int] | None] = [None] * num_workers
        any_skipped = False
        for i, worker in enumerate(available_workers):
            out_start = i * rows_per_worker
            out_end = min(out_start + rows_per_worker, H_out)
            if out_start >= H_out:
                any_skipped = True
                continue

            in_start_y_raw = out_start * dw_stride - dw_padding
            in_end_y_raw = (out_end - 1) * dw_stride + dw_kernel - dw_padding

            in_start_y = max(0, in_start_y_raw)
            in_end_y = min(H, in_end_y_raw)

            pad_top = in_start_y - in_start_y_raw
            pad_bottom = in_end_y_raw - in_end_y

            k_top, k_bottom = self._resolve_return_halo(return_halo_plan, i, out_end - out_start)

            halo_usable = (
                halo_state is not None
                and i < len(halo_state[0])
                and halo_state[0][i] is not None
            )
            if halo_usable:
                prev_start, prev_end = halo_state[0][i]
                ov_start = max(in_start_y, prev_start)
                ov_end = min(in_end_y, prev_end)
                if ov_end <= ov_start:
                    halo_usable = False
            if halo_usable:
                logger.debug(
                    f"[Distributor]: Reusing halo cache for worker {worker.worker_id} on block [{block.start_idx}-{block.end_idx}], "
                    f"halo rows {ov_start}-{ov_end} (pad_top={pad_top}, pad_bottom={pad_bottom})"
                )
                cache_use_start = ov_start - prev_start
                cache_use_end = ov_end - prev_start
                halo_top = feature_map[:, in_start_y:ov_start, :]
                halo_bottom = feature_map[:, ov_end:in_end_y, :]
                # payload = np.concatenate([halo_top, halo_bottom], axis=1)
                payload_bytes = (np.ascontiguousarray(halo_top).tobytes() + np.ascontiguousarray(halo_bottom).tobytes())
                payload = np.frombuffer(payload_bytes, dtype=np.uint8)
                task_msg = TaskMessage(
                    layer_type=block.layers[0].type,
                    layer_idx=block.start_idx,
                    end_layer_idx=block.end_idx,
                    in_channels=block.layers[0].in_channels,
                    in_h=in_end_y - in_start_y,
                    in_w=feature_map.shape[2],
                    out_channels=out_channels,
                    out_h=out_end - out_start,
                    out_w=W_out,
                    kernel_size=block.layers[0].kernel_size,
                    stride=block.layers[0].stride,
                    padding=0,
                    groups=block.layers[0].groups,
                    in_features=0,
                    out_features=0,
                    input_size=payload.size,
                    block_pad_top=pad_top,
                    block_pad_bottom=pad_bottom,
                    use_halo_cache=1,
                    halo_top_rows=halo_top.shape[1],
                    halo_bottom_rows=halo_bottom.shape[1],
                    cache_use_start=cache_use_start,
                    cache_use_end=cache_use_end,
                    prev_block_end_idx=halo_state[1],
                    return_halo=1 if (k_top or k_bottom) else 0,
                    return_halo_top_rows=k_top,
                    return_halo_bottom_rows=k_bottom,
                )
                send_full = block.layers[0].in_channels * (in_end_y - in_start_y) * feature_map.shape[2]
            else:
                input_patch = feature_map[:, in_start_y:in_end_y, :]
                payload = input_patch
                task_msg = TaskMessage(
                    layer_type=block.layers[0].type,
                    layer_idx=block.start_idx,
                    end_layer_idx=block.end_idx,
                    in_channels=block.layers[0].in_channels,
                    in_h=input_patch.shape[1],
                    in_w=input_patch.shape[2],
                    out_channels=out_channels,
                    out_h=out_end - out_start,
                    out_w=W_out,
                    kernel_size=block.layers[0].kernel_size,
                    stride=block.layers[0].stride,
                    padding=0,
                    groups=block.layers[0].groups,
                    in_features=0,
                    out_features=0,
                    input_size=input_patch.size,
                    block_pad_top=pad_top,
                    block_pad_bottom=pad_bottom,
                    return_halo=1 if (k_top or k_bottom) else 0,
                    return_halo_top_rows=k_top,
                    return_halo_bottom_rows=k_bottom,
                )
                send_full = None

            task = asyncio.create_task(self._send_task_to_worker(worker, task_msg, payload, current_layer_idx, send_full))
            tasks.append((worker, out_start, out_end, task, k_top, k_bottom))
            new_worker_rows[i] = (out_start, out_end)
            logger.debug(
                f"[Distributor]: Block [{block.start_idx}-{block.end_idx}] "
                f"assigned output rows {out_start}-{out_end} to worker {worker.worker_id}, "
                f"input patch shape {payload.shape}, "
                f"pad_top={pad_top}, pad_bottom={pad_bottom}, "
                f"{f', return halo rows top:{k_top} bottom:{k_bottom}' if (k_top or k_bottom) else ''}"
            )

        await asyncio.gather(*[t[3] for t in tasks])
        output_shape = (out_channels, H_out, W_out)
        output = await self._collect_results(tasks, output_shape, current_layer_idx)
        if any_skipped:
            return output, None
        return output, new_worker_rows
        # return await self._collect_results(tasks, output_shape, current_layer_idx)

    # ------------------------------------------------------------------
    # Shutdown helper (delegates to worker_manager)
    # ------------------------------------------------------------------

    async def shutdown_workers(self) -> None:
        """Send SHUTDOWN to every connected worker."""
        logger.info("[Distributor]: Sending shutdown message to all workers")
        shutdown_msg = b''
        for worker in self.worker_manager.workers.values():
            await self.worker_manager.send_message(worker, MessageType.SHUTDOWN, shutdown_msg)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _send_task_to_worker(self, worker: WorkerInfo, 
                                   task_msg: TaskMessage, input_patch: np.ndarray, 
                                   current_layer_idx: int,
                                   send_bytes_full: int | None = None) -> None:
        worker.state = WorkerState.BUSY
        input_bytes = np.ascontiguousarray(input_patch).tobytes()

        send_start = time.perf_counter()
        await self.worker_manager.send_message(worker, MessageType.TASK, task_msg.pack() + input_bytes)
        send_time = time.perf_counter() - send_start

        send_bytes_actual = len(input_bytes)
        if send_bytes_full is None:
            send_bytes_full = send_bytes_actual # no halo savings
        self.stats.record_worker_send(worker.worker_id, send_time * 1000, send_bytes_actual, send_bytes_full)

        logger.debug(
            f"[Distributor]: Sent task for layer {current_layer_idx} to worker {worker.worker_id}, waiting for result..."
            f"actual={send_bytes_actual}B, full={send_bytes_full}B"
        )

    async def _collect_results(self, tasks: list[tuple[WorkerInfo, int, int, asyncio.Task]], 
                               output_shape: tuple, current_layer_idx: int) -> np.ndarray:
        output = np.zeros(output_shape, dtype=np.uint8)
        logger.debug(
            f"[Distributor]: Collecting results from {len(tasks)} workers for layer {current_layer_idx}"
        )

        receive_tasks = []
        for worker, start_idx, end_idx, _, k_top, k_bottom in tasks:
            t = asyncio.create_task(
                self._receive_worker_result(worker, start_idx, end_idx, k_top, k_bottom, output, current_layer_idx)
            )
            receive_tasks.append(t)
        await asyncio.gather(*receive_tasks)

        return output

    async def _receive_worker_result(self, worker: WorkerInfo, 
                                     start_idx: int, end_idx: int, 
                                     k_top: int, k_bottom: int,
                                     output: np.ndarray, current_layer_idx: int) -> None:
        try:
            wait_start = time.perf_counter()
            header, payload = await self.worker_manager.receive_message(worker, timeout=60)
            wait_for_header_ms = (time.perf_counter() - wait_start) * 1000

            if not (header and payload):
                raise RuntimeError(f"Failed to receive result from worker {worker.worker_id}")

            if header.type == MessageType.ERROR:
                err_msg = ErrorMessage.unpack(payload)
                logger.error(
                    f"[Distributor]: Received error from worker {worker.worker_id}: "
                    f"error code: {err_msg.error_code}, message: {err_msg.description}"
                )
                raise RuntimeError(f"error: {err_msg.description}")

            if header.type != MessageType.RESULT:
                raise RuntimeError(f"Expected RESULT, got {header.type}")

            result_msg = ResultMessage.unpack(payload)
            logger.debug(f"[Distributor]: result message: {result_msg}")

            recv_start = time.perf_counter()
            output_data = await asyncio.wait_for(
                worker.reader.readexactly(result_msg.output_size), timeout=10
            )
            recv_time = time.perf_counter() - recv_start

            logger.debug(
                f"[Distributor]: Received result header from worker {worker.worker_id} "
                f"with output size {result_msg.output_size} bytes"
                f"(wait_header={wait_for_header_ms:.2f}ms)"
                f"(recv_time={recv_time*1000:.2f}ms)"
            )

            # Parse output data and write to correct position
            if output.ndim == 3:
                C, _, W = output.shape
                full_slice_bytes = (end_idx - start_idx) * C * W

                if k_top or k_bottom:
                    # halo return
                    expected =  C * (k_top + k_bottom) * W
                    if result_msg.output_size != expected:
                        raise RuntimeError(
                            f"Worker {worker.worker_id} return-halo size mismatched: "
                            f"got {result_msg.output_size} bytes, expected {expected} bytes "
                        )
                    top_bytes = C * k_top * W
                    if k_top:
                        top_patch = np.frombuffer(output_data[:top_bytes], dtype=np.uint8).reshape((C, k_top, W))
                        output[:, start_idx:start_idx+k_top, :] = top_patch
                    if k_bottom:
                        bottom_patch = np.frombuffer(output_data[top_bytes:], dtype=np.uint8).reshape((C, k_bottom, W))
                        output[:, end_idx-k_bottom:end_idx, :] = bottom_patch
                    logger.debug(
                        f"[Distributor]: Worker {worker.worker_id} returned halo rows top:{k_top} bottom:{k_bottom}, "
                        f"written to output rows [{start_idx}:{start_idx+k_top}, {end_idx-k_bottom}:{end_idx}] "
                    )
                else:
                    H_slice = end_idx - start_idx
                    output_patch = np.frombuffer(output_data, dtype=np.uint8).reshape((C, H_slice, W))
                    logger.debug(
                        f"[Distributor]: worker:{worker.worker_id}, output_data size: {len(output_data)} bytes, "
                        f"reshaped to {output_patch.shape}"
                    )
                    output[:, start_idx:end_idx, :] = output_patch
            else:
                full_slice_bytes = end_idx - start_idx
                output_patch = np.frombuffer(output_data, dtype=np.uint8)
                output[start_idx:end_idx] = output_patch

            # Update stats
            recv_bytes_actual = result_msg.output_size
            
            self.stats.record_worker_result(
                worker.worker_id,
                compute_ms=result_msg.compute_time_us / 1000,
                compress_ms=result_msg.compress_time_us / 1000,
                wait_header_ms=wait_for_header_ms,
                recv_time_ms=recv_time * 1000,
                recv_bytes_actual=recv_bytes_actual,
                recv_bytes_full=full_slice_bytes,
            )

            self.worker_manager.mark_worker_idle(worker)

            logger.debug(
                f"[Distributor]: Received result from worker {worker.worker_id}, "
                f"slice [{start_idx}, {end_idx}), "
                f"compute time: {result_msg.compute_time_us / 1000:.2f} ms, "
                f"compress time: {result_msg.compress_time_us / 1000:.2f} ms"
                f"wait_header={wait_for_header_ms:.2f}ms  "
                f"recv_body={recv_time * 1000:.2f}ms"
            )

        except Exception as e:
            logger.error(f"[Distributor]: Error receiving result from worker {worker.worker_id}: {e}")
            await self.shutdown_workers()
            worker.state = WorkerState.DISCONNECTED
            raise
    
    def _resolve_return_halo(self, return_halo_plan: tuple[list[int], list[int]] | None, worker_idx: int, slice_rows: int) -> tuple[int, int]:
        """ Resolve per-worker return halo rows """
        if return_halo_plan is None:
            return 0, 0
        k_top = return_halo_plan[0][worker_idx] if worker_idx < len(return_halo_plan[0]) else 0
        k_bottom = return_halo_plan[1][worker_idx] if worker_idx < len(return_halo_plan[1]) else 0
        # Ensure we don't request more halo rows than the slice size
        if k_top + k_bottom <= 0 or k_top + k_bottom >= slice_rows:
            return 0, 0
        return k_top, k_bottom
