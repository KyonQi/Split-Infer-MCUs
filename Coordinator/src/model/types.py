"""Pure data classes shared across the coordinator system."""

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np

# LayerType lives in protocol.py and is re-exported here for convenience so
# that model/* modules don't need to reach into protocol directly.
from ..protocol import LayerType


@dataclass
class LayerConfig:
    """Layer configuration for inference execution."""

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
    """Quantization parameters shared between coordinator and workers."""

    s_in: float
    z_in: int
    s_w: Union[float, np.ndarray]
    z_w: Union[float, np.ndarray]
    s_out: float
    z_out: int
    m: Union[float, np.ndarray]  # precomputed multiplier m = (s_in * s_w) / s_out
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
