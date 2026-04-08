"""Pluggable block-grouping strategies for different model architectures.

Each strategy takes a flat list of layers and produces a list of ``BlockConfig``
objects that the inference engine iterates over.  Swap to a different strategy to
support a new model architecture.

Usage
-----
``--mode block`` → uses the model-specific strategy (default: ``MobileNetV2Strategy``)
``--mode layer`` → forces ``SingleLayerStrategy`` (every layer executed individually)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from ..protocol import LayerType
from .types import BlockConfig, LayerConfig, QuantParams

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BlockGroupingStrategy(ABC):
    """Interface for grouping layers into execution blocks."""

    @abstractmethod
    def group(self, layers: list[LayerConfig], qps: list[QuantParams]) -> list[BlockConfig]:
        """Return an ordered list of execution blocks."""
        ...


# ---------------------------------------------------------------------------
# Concrete strategies
# ---------------------------------------------------------------------------

class MobileNetV2Strategy(BlockGroupingStrategy):
    """Recognise MobileNetV2 inverted-residual patterns.

    Patterns detected:
      • 3-layer block: expand (1×1) → depthwise (3×3) → project (1×1)
      • 2-layer block: depthwise (3×3) → project (1×1)   (e.g. blk0)
    Everything else becomes a single-layer block.
    """

    def group(self, layers: list[LayerConfig], qps: list[QuantParams]) -> list[BlockConfig]:
        blocks: list[BlockConfig] = []
        i = 0

        while i < len(layers):
            # ── Try 3-layer inverted-residual block ──
            if (
                i + 2 < len(layers)
                and layers[i].type in (LayerType.CONV, LayerType.POINTWISE)
                and layers[i].kernel_size == 1
                and layers[i + 1].type == LayerType.DEPTHWISE
                and layers[i + 2].type in (LayerType.CONV, LayerType.POINTWISE)
                and layers[i + 2].kernel_size == 1
            ):
                res_cache = layers[i].residual_add_to
                res_connect = layers[i + 2].residual_connect_from

                blocks.append(
                    BlockConfig(
                        start_idx=i,
                        end_idx=i + 2,
                        layers=layers[i : i + 3],
                        quant_params=qps[i : i + 3],
                        residual_cache_name=res_cache,
                        residual_connect_name=res_connect,
                    )
                )
                i += 3
                continue

            # ── Try 2-layer block: dw + proj ──
            if (
                i + 1 < len(layers)
                and layers[i].type == LayerType.DEPTHWISE
                and layers[i + 1].type in (LayerType.CONV, LayerType.POINTWISE)
                and layers[i + 1].kernel_size == 1
            ):
                blocks.append(
                    BlockConfig(
                        start_idx=i,
                        end_idx=i + 1,
                        layers=layers[i : i + 2],
                        quant_params=qps[i : i + 2],
                    )
                )
                i += 2
                continue

            # ── Single layer (init_conv, final_conv, fc_final, etc.) ──
            blocks.append(
                BlockConfig(
                    start_idx=i,
                    end_idx=i,
                    layers=[layers[i]],
                    quant_params=[qps[i]],
                )
            )
            i += 1

        return blocks


class MNASNet05Strategy(BlockGroupingStrategy):
    """Recognise MNASNet-0.5 patterns.

    Patterns detected:
      • 3-layer block: expand (1×1, stride=1) → depthwise → project (1×1)
        Same pattern as MobileNetV2 inverted-residual blocks.
      • 2-layer block: depthwise → project (1×1)
        Handles the MNASNet stem tail: init_dw + init_proj.  The leading
        init_conv (k=3, stride=2) must stay as a single layer because the
        block distributor assumes all pre-DW layers are 1×1/stride-1 and
        therefore do not change the spatial resolution — fusing a stride-2
        conv before the DW would make ``distribute_block`` compute a wrong
        output height and send misaligned input patches to workers.
    Everything else becomes a single-layer block.
    """

    def group(self, layers: list[LayerConfig], qps: list[QuantParams]) -> list[BlockConfig]:
        blocks: list[BlockConfig] = []
        i = 0

        while i < len(layers):
            # ── Try 3-layer block: 1×1 expand (stride=1) → depthwise → proj (1×1) ──
            # First layer must be kernel_size=1 so distribute_block's H_out
            # calculation (based solely on the DW params) stays correct.
            if (
                i + 2 < len(layers)
                and layers[i].type in (LayerType.CONV, LayerType.POINTWISE)
                and layers[i].kernel_size == 1
                and layers[i + 1].type == LayerType.DEPTHWISE
                and layers[i + 2].type in (LayerType.CONV, LayerType.POINTWISE)
                and layers[i + 2].kernel_size == 1
            ):
                res_cache = layers[i].residual_add_to
                res_connect = layers[i + 2].residual_connect_from

                blocks.append(
                    BlockConfig(
                        start_idx=i,
                        end_idx=i + 2,
                        layers=layers[i : i + 3],
                        quant_params=qps[i : i + 3],
                        residual_cache_name=res_cache,
                        residual_connect_name=res_connect,
                    )
                )
                i += 3
                continue

            # ── Try 2-layer block: dw + proj ──
            if (
                i + 1 < len(layers)
                and layers[i].type == LayerType.DEPTHWISE
                and layers[i + 1].type in (LayerType.CONV, LayerType.POINTWISE)
                and layers[i + 1].kernel_size == 1
            ):
                blocks.append(
                    BlockConfig(
                        start_idx=i,
                        end_idx=i + 1,
                        layers=layers[i : i + 2],
                        quant_params=qps[i : i + 2],
                    )
                )
                i += 2
                continue

            # ── Single layer (final_conv, fc_final, etc.) ──
            blocks.append(
                BlockConfig(
                    start_idx=i,
                    end_idx=i,
                    layers=[layers[i]],
                    quant_params=[qps[i]],
                )
            )
            i += 1

        return blocks


class MNASNet05HybridStrategy(BlockGroupingStrategy):
    """Adaptive block/layer strategy for MNASNet-0.5.

    Same overlap-ratio logic as ``HybridStrategy`` but uses
    ``MNASNet05Strategy`` as the candidate-block generator.  The stem
    init_conv (k=3, stride=2) remains a single layer; init_dw+init_proj
    form a 2-layer block subject to the usual overlap-ratio check.

    Parameters
    ----------
    num_workers:
        Number of MCU workers.
    max_overlap_ratio:
        Blocks whose overlap ratio exceeds this threshold fall back to
        layer-wise execution.
    """

    def __init__(self, num_workers: int = 4, max_overlap_ratio: float = 2) -> None:
        self.num_workers = num_workers
        self.max_overlap_ratio = max_overlap_ratio

    def group(self, layers: list[LayerConfig], qps: list[QuantParams]) -> list[BlockConfig]:
        candidate_blocks = MNASNet05Strategy().group(layers, qps)

        spatial_h = self._compute_spatial_heights(layers)

        final_blocks: list[BlockConfig] = []

        for blk in candidate_blocks:
            if blk.start_idx == blk.end_idx:
                final_blocks.append(blk)
                continue

            dw_layer = None
            for lyr in blk.layers:
                if lyr.type == LayerType.DEPTHWISE:
                    dw_layer = lyr
                    break

            if dw_layer is None:
                final_blocks.append(blk)
                continue

            h = spatial_h.get(dw_layer.layer_idx, 0)
            if h <= 0:
                final_blocks.append(blk)
                continue

            overlap = self._compute_overlap_ratio(h, dw_layer.kernel_size, dw_layer.stride, dw_layer.padding)

            if overlap > self.max_overlap_ratio:
                logger.info(
                    f"[MNASNet05HybridStrategy]: Splitting block [{blk.start_idx}-{blk.end_idx}] "
                    f"(overlap={overlap:.2f}x > {self.max_overlap_ratio}) into {len(blk.layers)} layers"
                )
                for lyr, qp in zip(blk.layers, blk.quant_params):
                    final_blocks.append(
                        BlockConfig(
                            start_idx=lyr.layer_idx,
                            end_idx=lyr.layer_idx,
                            layers=[lyr],
                            quant_params=[qp],
                            residual_cache_name=lyr.residual_add_to,
                            residual_connect_name=lyr.residual_connect_from,
                        )
                    )
            else:
                logger.debug(
                    f"[MNASNet05HybridStrategy]: Keeping block [{blk.start_idx}-{blk.end_idx}] "
                    f"(overlap={overlap:.2f}x <= {self.max_overlap_ratio})"
                )
                final_blocks.append(blk)

        return final_blocks

    def _compute_spatial_heights(self, layers: list[LayerConfig]) -> dict[int, int]:
        h = 224
        heights: dict[int, int] = {}
        for lyr in layers:
            heights[lyr.layer_idx] = h
            if lyr.type == LayerType.FC:
                h = 1
            else:
                h = (h + 2 * lyr.padding - lyr.kernel_size) // lyr.stride + 1
        return heights

    def _compute_overlap_ratio(self, h: int, kernel: int, stride: int, padding: int) -> float:
        h_out = (h + 2 * padding - kernel) // stride + 1
        n = self.num_workers
        rows_per_worker = -(-h_out // n)

        total_input_rows = 0
        for w in range(n):
            out_start = w * rows_per_worker
            out_end = min(out_start + rows_per_worker, h_out)
            if out_start >= h_out:
                break
            in_start_raw = out_start * stride - padding
            in_end_raw = (out_end - 1) * stride + kernel - padding
            in_start = max(0, in_start_raw)
            in_end = min(h, in_end_raw)
            total_input_rows += (in_end - in_start)

        return total_input_rows / h if h > 0 else 1.0


class SingleLayerStrategy(BlockGroupingStrategy):
    """Every layer is its own block — used with ``--mode layer`` for debugging."""

    def group(self, layers: list[LayerConfig], qps: list[QuantParams]) -> list[BlockConfig]:
        return [
            BlockConfig(
                start_idx=i,
                end_idx=i,
                layers=[layer],
                quant_params=[qps[i]],
                residual_cache_name=layer.residual_add_to,
                residual_connect_name=layer.residual_connect_from,
            )
            for i, layer in enumerate(layers)
        ]


class HybridStrategy(BlockGroupingStrategy):
    """Adaptive block/layer strategy based on per-block overlap ratio.

    For each candidate multi-layer block (expand + dw + project), computes the
    *overlap ratio* — the factor by which the expand layer's input rows are
    redundantly replicated across workers due to the DW kernel's spatial extent.

    When the ratio exceeds ``max_overlap_ratio`` the extra compute cost of the
    redundant expand calculation outweighs the communication savings, so the
    block is kept as individual layers (layer-wise).  Otherwise it is fused
    into a single block (block-wise).

    The ratio depends on ``num_workers``, the DW kernel size, and the spatial
    resolution at that point in the network::

        overlap_ratio = (sum of input rows each worker needs) / spatial_height

    Parameters
    ----------
    num_workers:
        Number of MCU workers (determines row-splitting granularity).
    max_overlap_ratio:
        Blocks whose overlap ratio exceeds this value fall back to layer-wise.
        Empirically 2.5 works well for 4 workers.
    """

    def __init__(self, num_workers: int = 4, max_overlap_ratio: float = 2) -> None:
        self.num_workers = num_workers
        self.max_overlap_ratio = max_overlap_ratio

    # ------------------------------------------------------------------ #

    def group(self, layers: list[LayerConfig], qps: list[QuantParams]) -> list[BlockConfig]:
        # First, build candidate blocks using the same pattern as MobileNetV2Strategy.
        candidate_blocks = MobileNetV2Strategy().group(layers, qps)

        # Compute spatial heights at each layer index so we can look them up.
        spatial_h = self._compute_spatial_heights(layers)

        final_blocks: list[BlockConfig] = []

        for blk in candidate_blocks:
            if blk.start_idx == blk.end_idx:
                # Single-layer block — nothing to decide.
                final_blocks.append(blk)
                continue

            # Find the DW layer inside this block.
            dw_layer = None
            for lyr in blk.layers:
                if lyr.type == LayerType.DEPTHWISE:
                    dw_layer = lyr
                    break

            if dw_layer is None:
                # No DW layer — keep as-is.
                final_blocks.append(blk)
                continue

            # Spatial height at the *DW layer input* (= expand output = same H).
            h = spatial_h.get(dw_layer.layer_idx, 0)
            if h <= 0:
                final_blocks.append(blk)
                continue

            overlap = self._compute_overlap_ratio(h, dw_layer.kernel_size, dw_layer.stride, dw_layer.padding)

            if overlap > self.max_overlap_ratio:
                # Explode this block back into individual single-layer blocks.
                logger.info(
                    f"[HybridStrategy]: Splitting block [{blk.start_idx}-{blk.end_idx}] "
                    f"(overlap={overlap:.2f}x > {self.max_overlap_ratio}) into {len(blk.layers)} layers"
                )
                for lyr, qp in zip(blk.layers, blk.quant_params):
                    final_blocks.append(
                        BlockConfig(
                            start_idx=lyr.layer_idx,
                            end_idx=lyr.layer_idx,
                            layers=[lyr],
                            quant_params=[qp],
                            residual_cache_name=lyr.residual_add_to,
                            residual_connect_name=lyr.residual_connect_from,
                        )
                    )
            else:
                logger.debug(
                    f"[HybridStrategy]: Keeping block [{blk.start_idx}-{blk.end_idx}] "
                    f"(overlap={overlap:.2f}x <= {self.max_overlap_ratio})"
                )
                final_blocks.append(blk)

        return final_blocks

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _compute_spatial_heights(self, layers: list[LayerConfig]) -> dict[int, int]:
        """Track spatial height through the network (input: 224×224)."""
        h = 224
        heights: dict[int, int] = {}
        for lyr in layers:
            heights[lyr.layer_idx] = h
            if lyr.type == LayerType.FC:
                h = 1
            else:
                h = (h + 2 * lyr.padding - lyr.kernel_size) // lyr.stride + 1
        return heights

    def _compute_overlap_ratio(self, h: int, kernel: int, stride: int, padding: int) -> float:
        """Compute how many times input rows are processed across all workers.

        Returns the ratio ``total_input_rows_across_workers / h``.
        A value of 1.0 means zero overlap (ideal); 4.0 means every worker
        processes the full height (worst case for 4 workers).
        """
        h_out = (h + 2 * padding - kernel) // stride + 1
        n = self.num_workers
        rows_per_worker = -(-h_out // n)  # ceil division

        total_input_rows = 0
        for w in range(n):
            out_start = w * rows_per_worker
            out_end = min(out_start + rows_per_worker, h_out)
            if out_start >= h_out:
                break
            in_start_raw = out_start * stride - padding
            in_end_raw = (out_end - 1) * stride + kernel - padding
            in_start = max(0, in_start_raw)
            in_end = min(h, in_end_raw)
            total_input_rows += (in_end - in_start)

        return total_input_rows / h if h > 0 else 1.0


# ---------------------------------------------------------------------------
# Strategy registry
# ---------------------------------------------------------------------------

STRATEGIES: dict[str, type[BlockGroupingStrategy]] = {
    'mobilenetv2': MobileNetV2Strategy,
    'mnasnet0_5': MNASNet05Strategy,
    'single_layer': SingleLayerStrategy,
    'hybrid': HybridStrategy,
    'mnasnet0_5_hybrid': MNASNet05HybridStrategy,
}

# Mapping from --mode CLI value to strategy key
_MODE_TO_STRATEGY: dict[str, str] = {
    'block': 'mobilenetv2',
    'layer': 'single_layer',
    'hybrid': 'hybrid',
    'mnasnet_block': 'mnasnet0_5',
    'mnasnet_hybrid': 'mnasnet0_5_hybrid',
}


def get_strategy(mode: str) -> BlockGroupingStrategy:
    """Instantiate a strategy from a ``--mode`` CLI value or a direct strategy key.

    Raises ``ValueError`` for unknown keys.
    """
    key = _MODE_TO_STRATEGY.get(mode, mode)
    cls = STRATEGIES.get(key)
    if cls is None:
        raise ValueError(
            f"Unknown execution mode / strategy '{mode}'. "
            f"Available: {list(STRATEGIES.keys())}"
        )
    return cls()
