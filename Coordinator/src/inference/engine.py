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

        for block in blocks:
            is_single = block.start_idx == block.end_idx

            if is_single:
                await self._execute_single_layer_block(block)
            else:
                await self._execute_multi_layer_block(block)

        total_time = time.time() - start_time
        logger.info(f"[Engine]: Inference completed in {total_time:.4f} seconds")
        self.stats.print_summary()

        return self.feature_map

    # ------------------------------------------------------------------
    # Block execution helpers
    # ------------------------------------------------------------------

    async def _execute_single_layer_block(self, block: BlockConfig) -> None:
        layer = block.layers[0]
        qp = block.quant_params[0]
        self.current_layer_idx = block.start_idx

        self.stats.begin_layer(
            layer_idx=block.start_idx,
            layer_name=layer.name,
            layer_type=LayerType(layer.type).name,
        )

        layer_start = time.perf_counter()
        await self._run_layer(layer, qp)
        layer_time = time.perf_counter() - layer_start

        self.stats.end_layer(layer_time * 1000)
        logger.debug(
            f"[Engine]: Layer {block.start_idx} done — "
            f"total={layer_time * 1000:.2f}ms"
        )

    async def _execute_multi_layer_block(self, block: BlockConfig) -> None:
        block_names = "+".join(l.name for l in block.layers)
        self.current_layer_idx = block.end_idx

        self.stats.begin_layer(
            layer_idx=f"{block.start_idx}-{block.end_idx}",
            layer_name=block_names,
            layer_type="BLOCK",
        )

        block_start = time.perf_counter()
        await self._run_block(block)
        block_time = time.perf_counter() - block_start

        self.stats.end_layer(block_time * 1000)
        logger.debug(
            f"[Engine]: Block [{block.start_idx}-{block.end_idx}] done — "
            f"total={block_time * 1000:.2f}ms"
        )

    # ------------------------------------------------------------------
    # Layer / block runners
    # ------------------------------------------------------------------

    async def _run_layer(self, layer: LayerConfig, quant_params: QuantParams) -> None:
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

        # Distribute
        if layer.type == LayerType.FC:
            self.feature_map = await self.distributor.distribute_fc(
                self.feature_map, layer, quant_params, self.current_layer_idx
            )
        else:
            self.feature_map = await self.distributor.distribute_conv(
                self.feature_map, layer, quant_params, self.current_layer_idx
            )

        # Apply residual connection
        if layer.residual_connect_from:
            self._apply_residual(layer.residual_connect_from)

    async def _run_block(self, block: BlockConfig) -> None:
        """Execute a multi-layer block in one round-trip.

        Residual handling stays on the coordinator side.
        """
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

        # Distribute block across workers
        self.feature_map = await self.distributor.distribute_block(
            self.feature_map, block, self.current_layer_idx
        )

        # Apply residual (add cached input to block output)
        if block.residual_connect_name:
            self.current_layer_idx = block.end_idx
            self._apply_residual(block.residual_connect_name)

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
