"""
Inference engine — owns the feature map and drives execution through blocks.

The engine is agnostic to *how* tasks are sent to workers (that's the
distributor's job) and to *how* layers are grouped (that's the strategy's job).
It simply iterates over the block list it receives and orchestrates residual
connections and statistics.
"""

from __future__ import annotations

import logging
import sys
import time
from typing import Optional

import numpy as np

from ..model.types import BlockConfig, LayerConfig, QuantParams
from ..protocol import LayerType
from ..stats import InferenceStats
from .distributor import TaskDistributor

logger = logging.getLogger(__name__)


class InferenceEngine:
    """Stateful inference runner.

    Holds the feature map and residual buffers for the duration of a single
    ``execute()`` call.
    """

    def __init__(self, distributor: TaskDistributor, stats: InferenceStats) -> None:
        self.distributor = distributor
        self.stats = stats

        # Per-inference state
        self.feature_map: Optional[np.ndarray] = None
        self.residual_buffers: dict[str, tuple[np.ndarray, float, int]] = {}
        self.current_layer_idx: int = 0

        # Needed for residual requantization (set before execute())
        self.quant_params_list: list[QuantParams] = []
        
        # Halo reuse
        self.use_halo = False
        self._prev_worker_rows: list[tuple[int, int]] | None = None
        self._prev_block_end_idx: int | None = None

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def execute(self, quantized_input: np.ndarray, 
                      layers: list[LayerConfig], qps: list[QuantParams], blocks: list[BlockConfig]) -> np.ndarray:
        """Run the full inference pipeline over *blocks*.

        Parameters
        ----------
        quantized_input:
            Already-quantized uint8 input feature map.
        layers / qps:
            Full flat layer and quant-param lists (needed for residual lookups).
        blocks:
            Execution blocks produced by a ``BlockGroupingStrategy``.
        """
        self.feature_map = quantized_input
        self.quant_params_list = qps
        self.residual_buffers.clear()
        # reset halo reuse state
        self._prev_worker_rows = None
        self._prev_block_end_idx = None


        logger.info(
            f"[Engine]: Starting inference — {len(layers)} layers in {len(blocks)} blocks, "
            f"input shape {self.feature_map.shape}"
        )
        for blk in blocks:
            names = [l.name for l in blk.layers]
            logger.debug(
                f"  Block [{blk.start_idx}-{blk.end_idx}]: {names}"
                f"  residual_cache={blk.residual_cache_name}  residual_connect={blk.residual_connect_name}"
            )

        start_time = time.time()

        for i, block in enumerate(blocks):
            next_block = blocks[i + 1] if i + 1 < len(blocks) else None
            is_single = block.start_idx == block.end_idx

            if is_single:
                await self._execute_single_layer_block(block, next_block)
            else:
                await self._execute_multi_layer_block(block, next_block)

        total_time = time.time() - start_time
        logger.info(f"[Engine]: Inference completed in {total_time:.4f} seconds")
        self.stats.print_summary()

        return self.feature_map

    # ------------------------------------------------------------------
    # Block execution helpers
    # ------------------------------------------------------------------

    async def _execute_single_layer_block(self, block: BlockConfig, next_block: BlockConfig | None) -> None:
        layer = block.layers[0]
        qp = block.quant_params[0]
        self.current_layer_idx = block.start_idx

        self.stats.begin_layer(
            layer_idx=block.start_idx,
            layer_name=layer.name,
            layer_type=LayerType(layer.type).name,
        )

        layer_start = time.perf_counter()
        await self._run_layer(layer, qp, block, next_block)
        layer_time = time.perf_counter() - layer_start

        self.stats.end_layer(layer_time * 1000)
        logger.debug(
            f"[Engine]: Layer {block.start_idx} done — "
            f"total={layer_time * 1000:.2f}ms"
        )

    async def _execute_multi_layer_block(self, block: BlockConfig, next_block: BlockConfig | None) -> None:
        block_names = "+".join(l.name for l in block.layers)
        self.current_layer_idx = block.end_idx

        self.stats.begin_layer(
            layer_idx=f"{block.start_idx}-{block.end_idx}",
            layer_name=block_names,
            layer_type="BLOCK",
        )

        block_start = time.perf_counter()
        await self._run_block(block, next_block)
        block_time = time.perf_counter() - block_start

        self.stats.end_layer(block_time * 1000)
        logger.debug(
            f"[Engine]: Block [{block.start_idx}-{block.end_idx}] done — "
            f"total={block_time * 1000:.2f}ms"
        )

    # ------------------------------------------------------------------
    # Layer / block runners
    # ------------------------------------------------------------------
    
    async def _run_layer(self, layer: LayerConfig, quant_params: QuantParams,
                         block: BlockConfig, next_block: BlockConfig | None) -> None:
        pre_start = time.perf_counter()
        # Cache residual if requested
        if layer.residual_add_to:
            self.residual_buffers[layer.residual_add_to] = (
                self.feature_map.copy(),
                quant_params.s_in,
                quant_params.z_in,
            )
            logger.debug(
                f"[Engine]: Stored residual buffer for {layer.residual_add_to} "
                f"with shape {self.feature_map.shape}"
            )

        # Global average pooling before FC
        if layer.type == LayerType.FC and self.feature_map.ndim == 3:
            gap_output = np.mean(self.feature_map, axis=(1, 2))
            self.feature_map = np.round(gap_output).astype(np.uint8)
            logger.debug(f"[Engine]: Applied GAP for FC layer, new shape {self.feature_map.shape}")
            with np.printoptions(threshold=sys.maxsize, linewidth=150):
                logger.debug(f"[Engine]: Sample GAP output values:\n{self.feature_map}\n")
        
        halo_state = None
        return_halo_plan = None
        if layer.type != LayerType.FC:
            if self.use_halo and self._prev_block_end_idx is not None:
                halo_state = (self._prev_worker_rows, self._prev_block_end_idx)
            if self.use_halo and next_block is not None:
                return_halo_plan = self._compute_return_halo_plan(block, next_block)
        
        self.stats.record_phase("engine_pre_ms", (time.perf_counter() - pre_start) * 1000)

        # Distribute
        if layer.type == LayerType.FC:
            self.feature_map = await self.distributor.distribute_fc(
                self.feature_map, layer, quant_params, self.current_layer_idx
            )
            new_worker_rows = None  # FC has no spatial halo
        else:
            self.feature_map, new_worker_rows = await self.distributor.distribute_conv(
                self.feature_map, layer, quant_params, self.current_layer_idx, 
                halo_state, return_halo_plan
            )

        post_start = time.perf_counter()
        # Apply residual connection
        if layer.residual_connect_from:
            self._apply_residual(layer.residual_connect_from)
            # invalidate the halo reuse state after residual connection
            self._prev_worker_rows = None
            self._prev_block_end_idx = None
        elif new_worker_rows is None:
            self._prev_worker_rows = None
            self._prev_block_end_idx = None
        else:
            self._prev_worker_rows = new_worker_rows
            self._prev_block_end_idx = self.current_layer_idx
        self.stats.record_phase("engine_post_ms", (time.perf_counter() - post_start) * 1000)


    async def _run_block(self, block: BlockConfig, next_block: BlockConfig | None) -> None:
        """Execute a multi-layer block in one round-trip.

        Residual handling stays on the coordinator side.
        """
        pre_start = time.perf_counter()
        # Save residual (block input before expand)
        if block.residual_cache_name:
            first_qp = block.quant_params[0]
            self.residual_buffers[block.residual_cache_name] = (
                self.feature_map.copy(),
                first_qp.s_in,
                first_qp.z_in,
            )
            logger.debug(
                f"[Engine]: Stored residual buffer '{block.residual_cache_name}' "
                f"with shape {self.feature_map.shape}"
            )
        
        # halo state
        halo_state = None
        if self.use_halo and self._prev_block_end_idx is not None:
            halo_state  = (self._prev_worker_rows, self._prev_block_end_idx)
        return_halo_plan = None
        if self.use_halo and next_block is not None:
            return_halo_plan = self._compute_return_halo_plan(block, next_block)
        self.stats.record_phase("engine_pre_ms", (time.perf_counter() - pre_start) * 1000)
        # Distribute block across workers
        self.feature_map, new_worker_rows = await self.distributor.distribute_block(
            self.feature_map, block, self.current_layer_idx, 
            halo_state, return_halo_plan
        )

        post_start = time.perf_counter()
        # Apply residual (add cached input to block output)
        if block.residual_connect_name:
            self.current_layer_idx = block.end_idx
            self._apply_residual(block.residual_connect_name)
            # invalidate the halo reuse state after residual connection
            self._prev_worker_rows = None
            self._prev_block_end_idx = None
        elif new_worker_rows is None:
            self._prev_worker_rows = None
            self._prev_block_end_idx = None
        else:
            self._prev_worker_rows = new_worker_rows
            self._prev_block_end_idx = block.end_idx
        self.stats.record_phase("engine_post_ms", (time.perf_counter() - post_start) * 1000)
    
    def _compute_return_halo_plan(self, block: BlockConfig, next_block: BlockConfig) -> Optional[dict[int, int]]:
        """ Decide whether current block's worker would return only boundary rows 
        Enable iff 
            1) next block will consume the output through forward halo
            2) doesn't require full feature map on the coordinator side
        """
        # next block must not cache residual
        if next_block.residual_cache_name:
            return None
        if next_block.residual_connect_name:
            return None
        for lyr in block.layers:
            if lyr.residual_add_to or lyr.residual_connect_from:
                return None
        # next block must not be FC / GAP
        if any(l.type == LayerType.FC for l in next_block.layers):
            return None
        # current block output must follows [C, H, W]
        if self.feature_map is None or self.feature_map.ndim != 3:
            return None
        
        H_curr_in = self.feature_map.shape[1]

        def _spatial_params(blk: BlockConfig) -> tuple[int, int, int] | None:
            if blk.start_idx == blk.end_idx:
                lyr = blk.layers[0]
                if lyr.type == LayerType.FC:
                    return None
                return lyr.kernel_size, lyr.stride, lyr.padding
            dw = next((l for l in blk.layers if l.type == LayerType.DEPTHWISE), None)
            if dw is None:
                return None
            return dw.kernel_size, dw.stride, dw.padding
        
        curr_p = _spatial_params(block)
        next_p = _spatial_params(next_block)
        if curr_p is None or next_p is None:
            return None
        
        curr_k, curr_s, curr_pad = curr_p
        next_k, next_s, next_pad = next_p
        H_curr_out = (H_curr_in + 2 * curr_pad - curr_k) // curr_s + 1
        H_next_out = (H_curr_out + 2 * next_pad - next_k) // next_s + 1

        num_workers = len(self.distributor.worker_manager.workers)
        if num_workers <= 1:
            return None
        
        curr_rows_per_worker = -(-H_curr_out // num_workers)  # ceiling division
        next_rows_per_worker = -(-H_next_out // num_workers)
        
        curr_rows: list[tuple[int, int] | None] = []
        next_in_rows: list[tuple[int, int] | None] = []
        for i in range(num_workers):
            a = i * curr_rows_per_worker
            b = min(a + curr_rows_per_worker, H_curr_out)
            curr_rows.append((a, b) if a < H_curr_out else None)

            na_out = i * next_rows_per_worker
            nb_out = min(na_out + next_rows_per_worker, H_next_out)
            if na_out >= H_next_out:
                next_in_rows.append(None)
                continue
            in_start_raw = na_out * next_s - next_pad
            in_end_raw = (nb_out - 1) * next_s + next_k - next_pad
            in_start = max(in_start_raw, 0)
            in_end = min(in_end_raw, H_curr_out)
            next_in_rows.append((in_start, in_end))
        
        # every worker should participate in next block's forward halo
        for i in range(num_workers):
            if curr_rows[i] is None or next_in_rows[i] is None:
                return None
            ca, cb = curr_rows[i]
            na, nb = next_in_rows[i]
            if min(cb, nb) - max(ca, na) <= 0:
                return None
        
        k_top = [0] * num_workers
        k_bottom = [0] * num_workers
        for i in range(num_workers):
            ca, cb = curr_rows[i]
            if i > 0:
                prev_cb = curr_rows[i - 1][1]
                prev_nb = next_in_rows[i - 1][1]
                k_top[i] = max(0, min(prev_nb - prev_cb, cb - ca))
            if i + 1 < num_workers:
                next_ca = curr_rows[i + 1][0]
                next_na = next_in_rows[i + 1][0]
                k_bottom[i] = max(0, min(next_ca - next_na, cb - ca))

        if all(k == 0 for k in k_top) and all(k == 0 for k in k_bottom):
            return None
        
        logger.debug(
            f"[Engine]: Computed return halo plan for block [{block.start_idx}-{block.end_idx}] -> "
            f"next block [{next_block.start_idx}-{next_block.end_idx}]: "
            f"curr_out_rows={curr_rows} next_in_rows={next_in_rows} "
            f"k_top={k_top} k_bottom={k_bottom}"
        )
        return k_top, k_bottom
            
    # ------------------------------------------------------------------
    # Residual connection
    # ------------------------------------------------------------------

    def _apply_residual(self, residual_from: str) -> None:
        if residual_from not in self.residual_buffers:
            logger.error(f"[Engine]: Residual buffer {residual_from} not found")
            return

        cached, res_s, res_zp = self.residual_buffers[residual_from]
        if cached.shape != self.feature_map.shape:
            logger.error(
                f"[Engine]: Residual buffer shape {cached.shape} != "
                f"feature map shape {self.feature_map.shape}"
            )
            return

        res_f = (cached.astype(np.float32) - res_zp) * res_s

        curr_scale = self.quant_params_list[self.current_layer_idx].s_out
        curr_zero_point = self.quant_params_list[self.current_layer_idx].z_out
        curr_f = (self.feature_map.astype(np.float32) - curr_zero_point) * curr_scale

        sum_f = curr_f + res_f
        target_s = self.quant_params_list[self.current_layer_idx].s_residual_out
        target_z = self.quant_params_list[self.current_layer_idx].z_residual_out
        self.feature_map = np.clip(
            np.round(sum_f / target_s + target_z), 0, 255
        ).astype(np.uint8)

        logger.debug(
            f"[Engine]: Applied residual connection from {residual_from} "
            f"to current layer {self.current_layer_idx}, feature map updated"
        )
