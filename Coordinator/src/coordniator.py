import asyncio
import logging
import socket
import sys
import time
import json
import numpy as np
from dataclasses import dataclass
from typing import Optional, Union
from .protocol import *
from .work_manager import *
from .rans import *
# from .task_queue import *

logger = logging.getLogger(__name__)

@dataclass
class LayerConfig:
    """layer config for inference execution"""
    name: str
    type: LayerType
    layer_idx: int
    in_channels: int
    out_channels: int
    kernel_size: int = 1
    stride: int = 1
    padding: int = 0
    groups: int = 1
    in_h: int = 0
    in_w: int = 0
    residual_add_to: Optional[str] = None
    residual_connect_from: Optional[str] = None

@dataclass
class QuantParams:
    """ quantization parameters needs to be shared between coordinator and workers """
    s_in: float
    z_in: int
    s_w: Union[float, np.ndarray]
    z_w: Union[float, np.ndarray]
    s_out: float
    z_out: int
    m: Union[float, np.ndarray] #float # precomputing multiplier for requantization m = (s_in * s_w) / s_out
    s_residual_out: Optional[float] = None
    z_residual_out: Optional[int] = None        

@dataclass
class BlockConfig:
    """A group of consecutive layers executed as one block on workers.
    
    For multi-layer blocks (e.g. expand + dw + project), all layers are
    computed on the MCU in a single round-trip, avoiding transfer of
    intermediate (expanded) feature maps.
    """
    start_idx: int                                    # first layer index (inclusive)
    end_idx: int                                      # last layer index (inclusive)
    layers: list                                      # LayerConfig objects in this block
    quant_params: list                                # QuantParams for each layer in the block
    residual_cache_name: Optional[str] = None         # save block input for residual add later
    residual_connect_name: Optional[str] = None       # add cached residual to block output

class Coordinator:
    def __init__(self, host: str = '192, 168, 1, 10', port: int = 54321):
        self.host: str = host
        self.port: int = port
        self.running = False
        self.worker_manager = WorkerManager()
        
        # inference managements
        self.feature_map: Optional[np.ndarray] = None
        self.residual_buffers: dict[str, tuple[np.ndarray, float, int]] = {}
        self.current_layer_idx: int = 0
        self.layer_config_list: list[LayerConfig] = [] # get the real vale by parsing the json file later
        self.quant_params_list: list[QuantParams] = [] # get the real value from calibration later

        # stats
        self.stats: list[dict] = []
        self.current_layer_stats: dict = {}

    async def start(self):
        self.running = True

        server = await asyncio.start_server(self.on_client_connected, self.host, self.port)
        logger.info(f"[Coordinator]: Coordinator started on {self.host}:{self.port}")

        tasks = [
            asyncio.create_task(server.serve_forever()), # start to listen
        ]
        try:
            await asyncio.gather(*tasks)
        except KeyboardInterrupt:
            logger.info("[Coordinator]: Shutting down coordinator...")
            self.running = False
            server.close()
            await server.wait_closed()
            logger.info("[Coordinator]: Coordinator stopped.")

    async def on_client_connected(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        """ connected callback """
        # Disable Nagle's algorithm and enable TCP_QUICKACK
        sock: socket.socket = writer.get_extra_info('socket')
        if sock is not None:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            # try:
            #     sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_QUICKACK, 1)
            # except (AttributeError, OSError):
            #     pass  # TCP_QUICKACK is Linux-only
            logger.info(f"[Coordinator]: Set TCP_NODELAY on socket")
        logger.info(f"[Coordinator]: New worker connected from {writer.get_extra_info('peername')}")
        worker = self.worker_manager.add_worker(reader, writer) # worker needs contains some info
        try:
            # once connected, worker will send a registration message
            # parse it and send corresponding ACK
            result: tuple[MessageHeader, bytes] = await self.worker_manager.receive_message(worker, timeout=2)
            if not result:
                logger.error("[Coordinator]: Failed to receive registration message from worker")
                self.worker_manager.remove_worker(worker)
                return
            
            # check header and payload
            header, payload = result
            if header.type != MessageType.REGISTER:
                logger.error(f"[Coordinator]: Expected REGISTER message, got {LayerType(header.type)}")
                self.worker_manager.remove_worker(worker)   
                return
            reg_msg = RegisterMessage.unpack(payload)
            worker.worker_id = header.worker_id # notice here we change to the real hardware assigned worker id after registration
            worker.clock_mhz = reg_msg.clock_mhz
            logger.info(f"[Coordinator]: Worker {worker.worker_id} registered with clock {worker.clock_mhz} MHz")

            # send ACK
            ack_msg = RegisterAckMessage(status=0, assigned_id=worker.worker_id)
            await self.worker_manager.send_message(worker, MessageType.REGISTER_ACK, ack_msg.pack())

            # TODO we need 3 steps handshake for better synchronization, but currently we just assume everything goes fine after registration

            # everything goes fine, worker -> idle, waiting for task assignment
            # go to event loop, waiting for messages from worker, which can be either RESULT or ERROR
            # worker.state = WorkerState.IDLE
            self.worker_manager.mark_worker_idle(worker)
            # await self.worker_event_loop(worker)
            
        except Exception as e:
            logger.error(f"[Coordinator]: Error handling for worker {worker.worker_id}: {e}")
            worker.state = WorkerState.DISCONNECTED
            self.worker_manager.remove_worker(worker)
        # finally:
        #     logger.info(f"[Coordinator]: Worker {worker.worker_id} disconnected")
        #     worker.state = WorkerState.DISCONNECTED
        #     self.worker_manager.remove_worker(worker)
    
    async def execute_inference(self, input_data: np.ndarray) -> np.ndarray:
        logger.info(f"[Coordinator]: Starting inference execution for input shape {input_data.shape}")

        self._parse_layer_configs() # parse the layer config and quant params from json file, and fill in the layer_config_list and quant_params_list
        self.feature_map = self._quantize_input(input_data, self.quant_params_list[0]) # quantize the input data to uint8, and fill in the feature_map
        self.residual_buffers.clear()
        
        blocks = self._group_layers_into_blocks()
        logger.info(f"[Coordinator]: Grouped {len(self.layer_config_list)} layers into {len(blocks)} blocks")
        for blk in blocks:
            names = [l.name for l in blk.layers]
            logger.debug(f"  Block [{blk.start_idx}-{blk.end_idx}]: {names}"
                         f"  residual_cache={blk.residual_cache_name}  residual_connect={blk.residual_connect_name}")
        
        start_time = time.time()
        for block in blocks:
            is_single = (block.start_idx == block.end_idx)

            if is_single:
                # ── Single-layer block: use original per-layer path ──
                layer = block.layers[0]
                qp = block.quant_params[0]
                self.current_layer_idx = block.start_idx

                self.current_layer_stats = {
                    "layer_idx": block.start_idx,
                    "layer_name": layer.name,
                    "layer_type": LayerType(layer.type).name,
                    "total_time_ms": 0.0,
                    "avg_compute_ms": 0.0,
                    "avg_compress_ms": 0.0,
                    "workers": {},
                }

                layer_start = time.perf_counter()
                await self._run_layer(layer, qp)
                layer_time = time.perf_counter() - layer_start

                self.current_layer_stats["total_time_ms"] = layer_time * 1000
                worker_stats = list(self.current_layer_stats["workers"].values())
                if worker_stats:
                    self.current_layer_stats["avg_compute_ms"] = float(np.mean([ws["mcu_compute_ms"] for ws in worker_stats]))
                    self.current_layer_stats["avg_compress_ms"] = float(np.mean([ws["mcu_compress_ms"] for ws in worker_stats]))
                self.stats.append(self.current_layer_stats)
                logger.debug(
                    f"[Coordinator]: Layer {block.start_idx} done — "
                    f"total={self.current_layer_stats['total_time_ms']:.2f}ms  "
                    f"compute={self.current_layer_stats['avg_compute_ms']:.2f}ms  "
                    f"compress={self.current_layer_stats['avg_compress_ms']:.2f}ms"
                )
            else:
                # ── Multi-layer block (e.g. expand + dw + project) ──
                block_names = "+".join(l.name for l in block.layers)
                self.current_layer_idx = block.end_idx  # residual uses the last layer's quant params

                self.current_layer_stats = {
                    "layer_idx": f"{block.start_idx}-{block.end_idx}",
                    "layer_name": block_names,
                    "layer_type": "BLOCK",
                    "total_time_ms": 0.0,
                    "avg_compute_ms": 0.0,
                    "avg_compress_ms": 0.0,
                    "workers": {},
                }

                block_start = time.perf_counter()
                await self._run_block(block)
                block_time = time.perf_counter() - block_start

                self.current_layer_stats["total_time_ms"] = block_time * 1000
                worker_stats = list(self.current_layer_stats["workers"].values())
                if worker_stats:
                    self.current_layer_stats["avg_compute_ms"] = float(np.mean([ws["mcu_compute_ms"] for ws in worker_stats]))
                    self.current_layer_stats["avg_compress_ms"] = float(np.mean([ws["mcu_compress_ms"] for ws in worker_stats]))
                self.stats.append(self.current_layer_stats)
                logger.debug(
                    f"[Coordinator]: Block [{block.start_idx}-{block.end_idx}] done — "
                    f"total={self.current_layer_stats['total_time_ms']:.2f}ms  "
                    f"compute={self.current_layer_stats['avg_compute_ms']:.2f}ms  "
                    f"compress={self.current_layer_stats['avg_compress_ms']:.2f}ms"
                )

        total_time = time.time() - start_time
        logger.info(f"[Coordinator]: Inference execution completed in {total_time:.4f} seconds")
        self.print_stats()

        return self.feature_map

    async def _run_layer(self, layer: LayerConfig, quant_params: QuantParams):
        if layer.residual_add_to:
            self.residual_buffers[layer.residual_add_to] = (self.feature_map.copy(), quant_params.s_in, quant_params.z_in)
            logger.debug(f"[Coordinator]: Stored residual buffer for {layer.residual_add_to} with shape {self.feature_map.shape}")
        
        # before fc, we needs a global average pooling and flatten
        if layer.type == LayerType.FC and self.feature_map.ndim == 3:
            gap_output = np.mean(self.feature_map, axis=(1, 2))
            self.feature_map = np.round(gap_output).astype(np.uint8)
            logger.debug(f"[Coordinator]: Applied GAP for FC layer, new shape {self.feature_map.shape}")
            with np.printoptions(threshold=sys.maxsize, linewidth=150):
                logger.debug(f"[Coordinator]: Sample GAP output values:\n{self.feature_map}\n")

        if layer.type == LayerType.FC:
            await self._distribute_fc(layer, quant_params)
        else:
            # deal with both conv2d and depthwise
            await self._distribute_conv(layer, quant_params)
        
        # apply residual
        if layer.residual_connect_from:
            await self._apply_residual(layer.residual_connect_from)
    
    async def _run_block(self, block: BlockConfig):
        """Execute a multi-layer block (e.g. expand + depthwise + project) in one round-trip.

        Residual handling stays on the coordinator side:
        - Before the block: save block input if residual_cache_name is set
        - After the block:  add cached residual if residual_connect_name is set
        """
        # Save residual (the block input, before expand)
        if block.residual_cache_name:
            first_qp = block.quant_params[0]
            self.residual_buffers[block.residual_cache_name] = (
                self.feature_map.copy(), first_qp.s_in, first_qp.z_in
            )
            logger.debug(f"[Coordinator]: Stored residual buffer '{block.residual_cache_name}' with shape {self.feature_map.shape}")

        # Distribute block across workers
        await self._distribute_block(block)

        # Apply residual (add cached input to block output)
        if block.residual_connect_name:
            self.current_layer_idx = block.end_idx
            await self._apply_residual(block.residual_connect_name)

    async def _distribute_block(self, block: BlockConfig):
        """Distribute a multi-layer block by row-splitting the block input.

        The coordinator sends un-padded input slices (with halo overlap for the
        DW receptive field) to each worker.  Each worker runs all layers in the
        block sequentially (expand → dw → project).

        DW padding is applied on the worker side *after* the expand layer,
        using the DW layer's own input zero-point.  This avoids the numerical
        error that arises when padding is applied before expand (where
        z_in_expand ≠ z_in_dw).

        The worker receives ``block_pad_top`` / ``block_pad_bottom`` in the task
        message to know how many rows of height-padding the DW layer needs.
        Width padding is always symmetric and derived from ``model_layer_config``.
        """
        C, H, W = self.feature_map.shape

        # Find the depthwise layer (the only spatial layer in the block)
        dw_layer = None
        for layer in block.layers:
            if layer.type == LayerType.DEPTHWISE:
                dw_layer = layer
                break

        if dw_layer is None:
            # No spatial layer — shouldn't happen for multi-layer blocks, fall back
            logger.warning(f"[Coordinator]: Block [{block.start_idx}-{block.end_idx}] has no DW layer, falling back to single-layer mode")
            for layer, qp in zip(block.layers, block.quant_params):
                self.current_layer_idx = layer.layer_idx
                await self._run_layer(layer, qp)
            return

        dw_padding = dw_layer.padding
        dw_kernel = dw_layer.kernel_size
        dw_stride = dw_layer.stride

        # Block output spatial dims (DW determines spatial transform; project is 1×1)
        H_out = (H + 2 * dw_padding - dw_kernel) // dw_stride + 1
        W_out = (W + 2 * dw_padding - dw_kernel) // dw_stride + 1

        # Block's final output channels
        out_channels = block.layers[-1].out_channels

        # Split output rows across workers
        available_workers = list(self.worker_manager.workers.values())
        num_workers = len(available_workers)
        rows_per_worker = int(np.ceil(H_out / num_workers))

        tasks = []
        for i, worker in enumerate(available_workers):
            out_start = i * rows_per_worker
            out_end = min(out_start + rows_per_worker, H_out)
            if out_start >= H_out:
                continue

            # Map DW output rows → un-padded block-input rows (with halo).
            # DW output row r reads expand-output rows [r*stride - dw_padding, r*stride - dw_padding + kernel).
            # Expand is 1×1 so these are the same block-input rows.
            in_start_y_raw = out_start * dw_stride - dw_padding
            in_end_y_raw   = (out_end - 1) * dw_stride + dw_kernel - dw_padding  # exclusive

            in_start_y = max(0, in_start_y_raw)
            in_end_y   = min(H, in_end_y_raw)

            # Height padding the worker must apply after expand, before DW
            pad_top    = in_start_y - in_start_y_raw   # >0 only for first chunk
            pad_bottom = in_end_y_raw - in_end_y       # >0 only for last chunk

            input_patch = self.feature_map[:, in_start_y:in_end_y, :]

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
                padding=0,  # no coordinator-side padding
                groups=block.layers[0].groups,
                in_features=0,
                out_features=0,
                input_size=input_patch.size,
                block_pad_top=pad_top,
                block_pad_bottom=pad_bottom,
            )

            task = asyncio.create_task(
                self._send_task_to_worker(worker, task_msg, input_patch)
            )
            tasks.append((worker, out_start, out_end, task))
            logger.debug(
                f"[Coordinator]: Block [{block.start_idx}-{block.end_idx}] "
                f"assigned output rows {out_start}-{out_end} to worker {worker.worker_id}, "
                f"input patch shape {input_patch.shape}, "
                f"pad_top={pad_top}, pad_bottom={pad_bottom}"
            )

        await asyncio.gather(*[t[3] for t in tasks])
        output_shape = (out_channels, H_out, W_out)
        self.feature_map = await self._collect_results(tasks, output_shape)
    

    async def _distribute_conv(self, layer: LayerConfig, quant_params: QuantParams):
        """Split the feature map by rows"""
        C, H, W = self.feature_map.shape
        H_out = (H + 2 * layer.padding - layer.kernel_size) // layer.stride + 1
        W_out = (W + 2 * layer.padding - layer.kernel_size) // layer.stride + 1
        
        if layer.padding > 0:
            padded = np.pad(
                self.feature_map,
                ((0, 0), (layer.padding, layer.padding), (layer.padding, layer.padding)),
                mode='constant',
                constant_values=quant_params.z_in
            )
        else:
            padded = self.feature_map
        
        available_workers = list(self.worker_manager.workers.values())
        num_workers = len(available_workers) # TODO maybe get idle workers
        rows_per_worker = int(np.ceil(H_out / num_workers))
        tasks = []
        
        for i, worker in enumerate(available_workers):
            start_row = i * rows_per_worker
            end_row = min(start_row + rows_per_worker, H_out)

            if start_row >= H_out:
                continue

            in_start_y = start_row * layer.stride
            in_end_y = (end_row - 1) * layer.stride + layer.kernel_size
            input_patch = padded[:, in_start_y:in_end_y, :]

            task_msg = TaskMessage(
                layer_type=layer.type,
                layer_idx=self.current_layer_idx,
                end_layer_idx=self.current_layer_idx,  # single layer
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
                input_size=input_patch.size
            )

            task = asyncio.create_task(
                self._send_task_to_worker(worker, task_msg, input_patch)
            )
            tasks.append((worker, start_row, end_row, task))
            logger.debug(f"[Coordinator]: Assigned output rows {start_row}-{end_row} to worker {worker.worker_id} for layer {layer.name}")
        
        await asyncio.gather(*[t[3] for t in tasks])
        output_shape = (layer.out_channels, H_out, W_out)
        self.feature_map = await self._collect_results(tasks, output_shape)

    async def _distribute_fc(self, layer: LayerConfig, quant_params: QuantParams):
        """Split the feature map by output classes"""
        input_vec = self.feature_map.flatten()
        total_classes = layer.out_channels
        available_workers = list(self.worker_manager.workers.values())
        num_workers = len(available_workers) # TODO maybe get idle workers
        classes_per_worker = int(np.ceil(total_classes / num_workers))

        logger.debug(f"[Coordinator]: Distributing FC layer {layer.name} with {total_classes} classes across {num_workers} workers")
        
        tasks = []
        for i, worker in enumerate(available_workers):
            # start_cls = i * classes_per_worker
            # end_cls = min(start_cls + classes_per_worker, total_classes)
            start_cls = worker.worker_id * classes_per_worker
            end_cls = min(start_cls + classes_per_worker, total_classes)
            
            if start_cls >= total_classes:
                continue
            
            task_msg = TaskMessage(
                layer_type=layer.type,
                layer_idx=self.current_layer_idx,
                end_layer_idx=self.current_layer_idx,  # single layer
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
                input_size=input_vec.size
            )
            task = asyncio.create_task(
                self._send_task_to_worker(worker, task_msg, input_vec)
            )
            tasks.append((worker, start_cls, end_cls, task))
            logger.debug(f"[Coordinator]: Assigned classes {start_cls}-{end_cls} to worker {worker.worker_id} for FC layer {layer.name}")
        await asyncio.gather(*[t[3] for t in tasks])
        
        output_shape = (total_classes,)
        self.feature_map = await self._collect_results(tasks, output_shape)
        
    # TODO need further check
    async def _apply_residual(self, residual_from: str):
        if residual_from not in self.residual_buffers:
            logger.error(f"[Coordinator]: Residual buffer {residual_from} not found for residual connection")
            return
        
        cached, res_s, res_zp = self.residual_buffers[residual_from]
        if cached.shape != self.feature_map.shape:
            logger.error(f"[Coordinator]: Residual buffer shape {cached.shape} does not match current feature map shape {self.feature_map.shape}")
            return
        
        res_f = (cached.astype(np.float32) - res_zp) * res_s

        curr_scale = self.quant_params_list[self.current_layer_idx].s_out
        curr_zero_point = self.quant_params_list[self.current_layer_idx].z_out
        curr_f = (self.feature_map.astype(np.float32) - curr_zero_point) * curr_scale
        
        sum_f = curr_f + res_f
        target_s = self.quant_params_list[self.current_layer_idx].s_residual_out
        target_z = self.quant_params_list[self.current_layer_idx].z_residual_out
        self.feature_map = np.clip(np.round(sum_f / target_s + target_z), 0, 255).astype(np.uint8)

        logger.debug(f"[Coordinator]: Applied residual connection from {residual_from} to current layer {self.current_layer_idx}, feature map updated")

        if self.current_layer_idx == 51:
            logger.debug(f"[Coordinator]: Completed layer {residual_from}, output feature map shape: {self.feature_map.shape}")
            # hex_str = np.array2string(
            #     self.feature_map[1, :, :], 
            #     formatter={'int': lambda x: f'0x{x:02X}'}
            # )

            # logger.debug(f"[Coordinator]: Input for this layer is:\n{padded[1, :1, :]}\n")
            logger.debug(f"[Coordinator]: Sample output hex values:\n{self.feature_map[1, :2, :]}\n")

    async def _send_task_to_worker(self, worker: WorkerInfo, task_msg: TaskMessage, input_patch: np.ndarray):
        worker.state = WorkerState.BUSY
        # Ensure C-contiguous layout before serializing: slicing along axis-1 (e.g. padded[:, a:b, :])
        input_bytes = np.ascontiguousarray(input_patch).tobytes()

        # # rANS compress input data before sending
        # original_size = len(input_bytes)
        # compressed_bytes = rans_compress(input_bytes)
        # if len(compressed_bytes) < original_size:
        #     # Compression was beneficial — update task_msg.input_size to compressed size
        #     task_msg.input_size = len(compressed_bytes)
        #     input_bytes = compressed_bytes
        #     logger.debug(
        #         f"[Coordinator]: Compressed input for worker {worker.worker_id}: "
        #         f"{original_size} -> {len(compressed_bytes)} bytes "
        #         f"({original_size / len(compressed_bytes):.2f}x)"
        #     )

        send_start = time.perf_counter()
        await self.worker_manager.send_message(worker, MessageType.TASK, task_msg.pack() + input_bytes)
        send_time = time.perf_counter() - send_start

        # init the worker's stats
        self.current_layer_stats["workers"][worker.worker_id] = {
            "send_time_ms": send_time * 1000,
            "recv_time_ms": 0.0,
            "mcu_compute_ms": 0.0,
        }

        logger.debug(f"[Coordinator]: Sent task for layer {self.current_layer_idx} to worker {worker.worker_id}, waiting for result...")

    async def _collect_results(self, tasks: list[asyncio.Task], output_shape: tuple) -> np.ndarray:
        output = np.zeros(output_shape, dtype=np.uint8)
        num_workers = len(tasks)
        logger.debug(f"[Coordinator]: Collecting results from {num_workers} workers for layer {self.current_layer_idx}")
        
        receive_tasks = []
        for worker, start_idx, end_idx, _ in tasks:
            task = asyncio.create_task(
                self._receive_worker_result(worker, start_idx, end_idx, output)
            )
            receive_tasks.append(task)
        await asyncio.gather(*receive_tasks)
        
        return output
    
    async def _receive_worker_result(self, worker: WorkerInfo, start_idx: int, end_idx: int, output: np.ndarray):
        try:
            #  wait for result message
            header, payload = await self.worker_manager.receive_message(
                worker, 
                timeout=60
            )
            
            if not (header and payload):
                raise RuntimeError(f"Failed to receive result from worker {worker.worker_id}")
            
            if header.type == MessageType.ERROR:
                err_msg = ErrorMessage.unpack(payload)
                logger.error(f"[Coordinator]: Received error from worker {worker.worker_id}: error code: {err_msg.error_code}, message: {err_msg.description}")
                raise RuntimeError(f"error: {err_msg.description}")
            
            if header.type != MessageType.RESULT:
                raise RuntimeError(f"Expected RESULT, got {LayerType(header.type)}")
            
            result_msg = ResultMessage.unpack(payload)
            logger.debug(f"[Coordinator]: result message: {result_msg}")
            
            # read exact output data
            # output_data = await worker.reader.readexactly(result_msg.output_size)
            recv_start = time.perf_counter()
            output_data = await asyncio.wait_for(worker.reader.readexactly(result_msg.output_size), timeout=10)
            recv_time = time.perf_counter() - recv_start

            logger.debug(f"[Coordinator]: Received result header from worker {worker.worker_id} with output size {result_msg.output_size} bytes")
            
            # # rANS decompression
            # if is_rans_compressed(output_data):
            #     decompress_start = time.perf_counter()
            #     raw_bytes = rans_decompress(output_data)
            #     decompress_time = time.perf_counter() - decompress_start
            #     logger.debug(
            #         f"[Coordinator]: Decompressed rANS from worker {worker.worker_id}: "
            #         f"{len(output_data)} -> {len(raw_bytes)} bytes, "
            #         f"decompress time: {decompress_time * 1000:.2f} ms"
            #     )
            #     output_data = raw_bytes


            # parse output data and write to the correct position in the output feature map
            if output.ndim == 3:
                # Conv layer: (C, H_slice, W)
                C, _, W = output.shape
                H_slice = end_idx - start_idx
                output_patch = np.frombuffer(output_data, dtype=np.uint8).reshape(
                    (C, H_slice, W)
                )
                logger.debug(f"[Coordinator]: worker:{worker.worker_id}, output_data size: {len(output_data)} bytes, reshaped to {output_patch.shape}")
                output[:, start_idx:end_idx, :] = output_patch
            else:
                # Linear layer: (num_classes,)
                output_patch = np.frombuffer(output_data, dtype=np.uint8)
                output[start_idx:end_idx] = output_patch
            
            # update stats
            # self.stats.total_comm_volume += result_msg.output_size
            # self.stats.total_compute_time += result_msg.compute_time_us / 1e6
            if worker.worker_id in self.current_layer_stats["workers"]:
                ws = self.current_layer_stats["workers"][worker.worker_id]
                ws["mcu_compute_ms"] = result_msg.compute_time_us / 1000
                ws["mcu_compress_ms"] = result_msg.compress_time_us / 1000
                ws["recv_time_ms"] = recv_time * 1000
            
            # mark worker idle again
            # worker.state = WorkerState.IDLE
            self.worker_manager.mark_worker_idle(worker)
            
            logger.debug(f"[Coordinator]: Received result from worker {worker.worker_id}, "
                        f"slice [{start_idx}, {end_idx}), "
                        f"compute time: {result_msg.compute_time_us / 1000:.2f} ms, "
                        f"compress time: {result_msg.compress_time_us / 1000:.2f} ms")
        
        except Exception as e:
            logger.error(f"[Coordinator]: Error receiving result from worker {worker.worker_id}: {e}")
            await self.shutdown_workers() # if any error happens, we shutdown all workers to avoid hanging
            worker.state = WorkerState.DISCONNECTED
            raise
    
    async def shutdown_workers(self):
        logger.info(f"[Coordinator]: Sending shutdown message to all workers")
        shutdown_msg = b'' # no payload needed for shutdown
        for worker in self.worker_manager.workers.values():
            await self.worker_manager.send_message(worker, MessageType.SHUTDOWN, shutdown_msg)
        #  await self.worker_manager.send_message(worker, MessageType.TASK, task_msg.pack() + input_patch.tobytes())

    def _group_layers_into_blocks(self) -> list[BlockConfig]:
        """Group consecutive layers into blocks for block-level execution.

        Recognises two patterns:
          • 3-layer block: expand (1×1 conv) → depthwise (3×3) → project (1×1 conv)
          • 2-layer block: depthwise (3×3) → project (1×1 conv)   (e.g. blk0)
        Everything else becomes a single-layer block.
        """
        blocks: list[BlockConfig] = []
        layers = self.layer_config_list
        qps = self.quant_params_list
        i = 0

        while i < len(layers):
            # ── Try 3-layer inverted-residual block ──
            if (i + 2 < len(layers)
                and layers[i].type in (LayerType.CONV, LayerType.POINTWISE)
                and layers[i].kernel_size == 1
                and layers[i+1].type == LayerType.DEPTHWISE
                and layers[i+2].type in (LayerType.CONV, LayerType.POINTWISE)
                and layers[i+2].kernel_size == 1):

                res_cache = layers[i].residual_add_to
                res_connect = layers[i+2].residual_connect_from

                blocks.append(BlockConfig(
                    start_idx=i, end_idx=i + 2,
                    layers=layers[i:i+3], quant_params=qps[i:i+3],
                    residual_cache_name=res_cache,
                    residual_connect_name=res_connect,
                ))
                i += 3
                continue

            # ── Try 2-layer block: dw + proj ──
            if (i + 1 < len(layers)
                and layers[i].type == LayerType.DEPTHWISE
                and layers[i+1].type in (LayerType.CONV, LayerType.POINTWISE)
                and layers[i+1].kernel_size == 1):

                blocks.append(BlockConfig(
                    start_idx=i, end_idx=i + 1,
                    layers=layers[i:i+2], quant_params=qps[i:i+2],
                ))
                i += 2
                continue

            # ── Single layer (init_conv, final_conv, fc_final, etc.) ──
            blocks.append(BlockConfig(
                start_idx=i, end_idx=i,
                layers=[layers[i]], quant_params=[qps[i]],
            ))
            i += 1

        return blocks


    def _parse_layer_configs(self, json_path: str = './src/model_config.json'):
        with open(json_path, 'r') as f:
            data = json.load(f)
        logger.info(f"[Coordinator]: Loaded model config from {json_path}, total layers: {len(data['layers'])}")
        
        layer_configs = []
        quant_params = []
        for idx, layer_data in enumerate(data["layers"]):
            layer_config_dict = layer_data["layer_config"]
            quant_params_dict = layer_data["quant_params"]

            layer_type = LayerType(layer_config_dict["type"])

            cfg = LayerConfig(
                name=layer_config_dict["name"],
                type=layer_type,
                layer_idx=idx,
                in_channels=layer_config_dict["in_channels"],
                out_channels=layer_config_dict["out_channels"],
                kernel_size=layer_config_dict["kernel_size"],
                stride=layer_config_dict["stride"],
                padding=layer_config_dict["padding"],
                groups=layer_config_dict["groups"],
                residual_add_to=layer_config_dict["residual_add_to"],
                residual_connect_from=layer_config_dict["residual_connect_from"]
            )

            qp = QuantParams(
                s_in=float(quant_params_dict["s_in"]),
                z_in=int(quant_params_dict["z_in"]),
                s_w=np.array(quant_params_dict["s_w"], dtype=np.float32),
                z_w=np.array(quant_params_dict["z_w"], dtype=np.int32),
                s_out=float(quant_params_dict["s_out"]),
                z_out=int(quant_params_dict["z_out"]),
                m=np.array(quant_params_dict["m"], dtype=np.float32),
                s_residual_out=float(quant_params_dict["s_residual_out"]) if quant_params_dict["s_residual_out"] is not None else None,
                z_residual_out=int(quant_params_dict["z_residual_out"]) if quant_params_dict["z_residual_out"] is not None else None
            )

            layer_configs.append(cfg)
            quant_params.append(qp)

        self.layer_config_list = layer_configs
        self.quant_params_list = quant_params
        logger.info(f"[Coordinator]: Parsed {len(self.layer_config_list)} layers and quantization parameters from config")
    
    def _quantize_input(self, input_data: np.ndarray, quant_params: QuantParams) -> np.ndarray:
        s_in = quant_params.s_in
        z_in = quant_params.z_in
        quantized = np.clip(np.round(input_data / s_in + z_in), 0, 255).astype(np.uint8)
        return quantized
    
    def print_stats(self):
        logger.info(f"[Coordinator]: Inference execution stats:")
        for s in self.stats:
            idx_str = str(s['layer_idx'])
            logger.info(
                f"Layer {idx_str:>5} [{s['layer_type']:>8}] {s['layer_name']}: "
                f"total={s['total_time_ms']:.2f}ms  "
                f"compute={s.get('avg_compute_ms', 0):.2f}ms  "
                f"compress={s.get('avg_compress_ms', 0):.2f}ms"
            )