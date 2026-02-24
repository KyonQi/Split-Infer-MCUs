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


# ---------------------------------------------------------------------------
# Strategy registry
# ---------------------------------------------------------------------------

STRATEGIES: dict[str, type[BlockGroupingStrategy]] = {
    'mobilenetv2': MobileNetV2Strategy,
    'single_layer': SingleLayerStrategy,
    # Future: 'resnet': ResNetStrategy, ...
}

# Mapping from --mode CLI value to strategy key
_MODE_TO_STRATEGY: dict[str, str] = {
    'block': 'mobilenetv2',
    'layer': 'single_layer',
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
