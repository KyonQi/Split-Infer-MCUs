"""Model configuration loading and input quantization — pure functions, no state."""

import json
import logging
from typing import Optional

import numpy as np

from ..protocol import LayerType
from .types import LayerConfig, QuantParams

logger = logging.getLogger(__name__)


def parse_layer_configs(
    json_path: str = './src/model_config.json',
) -> tuple[list[LayerConfig], list[QuantParams]]:
    """Parse model layer configurations and quantization parameters from a JSON file.

    Returns:
        (layer_configs, quant_params) — parallel lists, one entry per layer.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    logger.info(f"Loaded model config from {json_path}, total layers: {len(data['layers'])}")

    layer_configs: list[LayerConfig] = []
    quant_params: list[QuantParams] = []

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
            residual_add_to=layer_config_dict.get("residual_add_to"),
            residual_connect_from=layer_config_dict.get("residual_connect_from"),
        )

        qp = QuantParams(
            s_in=float(quant_params_dict["s_in"]),
            z_in=int(quant_params_dict["z_in"]),
            s_w=np.array(quant_params_dict["s_w"], dtype=np.float32),
            z_w=np.array(quant_params_dict["z_w"], dtype=np.int32),
            s_out=float(quant_params_dict["s_out"]),
            z_out=int(quant_params_dict["z_out"]),
            m=np.array(quant_params_dict["m"], dtype=np.float32),
            s_residual_out=(
                float(quant_params_dict["s_residual_out"])
                if quant_params_dict.get("s_residual_out") is not None
                else None
            ),
            z_residual_out=(
                int(quant_params_dict["z_residual_out"])
                if quant_params_dict.get("z_residual_out") is not None
                else None
            ),
        )

        layer_configs.append(cfg)
        quant_params.append(qp)

    logger.info(f"Parsed {len(layer_configs)} layers and quantization parameters from config")
    return layer_configs, quant_params


def quantize_input(input_data: np.ndarray, quant_params: QuantParams) -> np.ndarray:
    """Quantize floating-point input to uint8 using the first layer's quant params."""
    s_in = quant_params.s_in
    z_in = quant_params.z_in
    quantized = np.clip(np.round(input_data / s_in + z_in), 0, 255).astype(np.uint8)
    return quantized
