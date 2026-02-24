# Coordinator — Distributed DNN Inference on MCU Clusters

A Python-based coordinator that orchestrates quantized deep neural network inference across a cluster of MCU (microcontroller) worker nodes over TCP. The coordinator splits feature maps, dispatches computation tasks to workers, collects results, and assembles the final output — enabling models like MobileNetV2 to run across resource-constrained devices.

## Overview

```
┌──────────────────────────────────────────────────────┐
│                    Coordinator (PC)                   │
│                                                       │
│  main.py ─► Coordinator ─► InferenceEngine            │
│                               │                       │
│                         TaskDistributor               │
│                        ┌──────┼──────┐                │
│                        ▼      ▼      ▼                │
│                     Worker  Worker  Worker   ...      │
│                     (MCU)   (MCU)   (MCU)             │
└──────────────────────────────────────────────────────┘
```

The coordinator:

1. **Accepts TCP connections** from MCU workers and registers them.
2. **Loads a model configuration** (layer definitions + quantization parameters) from a JSON file.
3. **Groups layers** into execution blocks using a pluggable strategy (e.g. fuse `expand → depthwise → project` into a single round-trip).
4. **Splits feature maps** by rows (conv/depthwise) or output classes (FC) and distributes slices to workers.
5. **Collects partial results**, reassembles the full output, and handles residual connections on the coordinator side.
6. **Reports per-layer timing statistics** (total latency, MCU compute time, communication overhead).

## Project Structure

```
Coordinator/
├── main.py                         # CLI entry point
├── pyproject.toml                  # Project metadata & dependencies
├── data/
│   ├── panda.jpg                   # Example input image
│   └── imagenet_labels.json        # ImageNet class labels
├── src/
│   ├── config.py                   # CoordinatorConfig dataclass
│   ├── coordinator.py              # Thin TCP server + orchestration shell
│   ├── protocol.py                 # Binary wire protocol (header, task, result)
│   ├── work_manager.py             # Worker connection lifecycle
│   ├── stats.py                    # Per-layer performance statistics
│   ├── rans.py                     # rANS entropy codec (C FFI + Python fallback)
│   ├── model/
│   │   ├── types.py                # LayerConfig, QuantParams, BlockConfig
│   │   ├── loader.py               # JSON config parser + input quantization
│   │   └── block_strategy.py       # Pluggable layer-grouping strategies
│   └── inference/
│       ├── engine.py               # Inference state machine (feature map, residuals)
│       └── distributor.py          # Task splitting & worker communication
├── tests/
│   └── test_coordinator_core.py    # Unit tests for all modules
└── img/
    └── plot_layer_analysis.py      # Visualization utilities
```

### Module Responsibilities

| Module | Role |
|---|---|
| `config.py` | Central `CoordinatorConfig` dataclass — built from CLI args, threaded through the system |
| `coordinator.py` | TCP server lifecycle, worker registration handshake, delegates inference to engine |
| `model/types.py` | Pure data classes: `LayerConfig`, `QuantParams`, `BlockConfig` |
| `model/loader.py` | Parses `model_config.json`, returns layer configs + quant params; input quantization |
| `model/block_strategy.py` | `BlockGroupingStrategy` ABC + concrete strategies (`MobileNetV2Strategy`, `SingleLayerStrategy`) |
| `inference/engine.py` | Owns the feature map & residual buffers; iterates over blocks; applies GAP before FC |
| `inference/distributor.py` | Splits data across workers, sends tasks, collects & reassembles results |
| `stats.py` | Tracks send/recv/compute/compress times per layer; prints summary |
| `protocol.py` | Binary message definitions (little-endian `struct` packing) for the custom TCP protocol |
| `work_manager.py` | Manages worker connections, send/receive with timeouts, idle queue |
| `rans.py` | Optional rANS entropy compression/decompression (C shared library with Python fallback) |

## Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.8 (for image preprocessing)
- NumPy ≥ 2.2
- Pillow ≥ 12.1

Install with [uv](https://github.com/astral-sh/uv):

```bash
uv sync
```

Or with pip:

```bash
pip install torch torchvision numpy pillow
```

## Usage

### Basic Run

```bash
python main.py --workers 4
```

The coordinator binds to `192.168.1.10:54321` by default, waits for 4 MCU workers to connect, then runs inference on `data/panda.jpg` and prints the top-5 ImageNet predictions.

### CLI Options

```
python main.py [OPTIONS]
```

| Option | Default | Description |
|---|---|---|
| `--workers N` | `4` | Number of MCU workers to wait for before starting inference |
| `--mode {block,layer}` | `block` | Execution mode (see below) |
| `--model-config PATH` | `./src/model_config.json` | Path to the model configuration JSON |
| `--host ADDR` | `192.168.1.10` | Coordinator TCP bind address |
| `--port PORT` | `54321` | Coordinator TCP bind port |
| `--log-level LEVEL` | `INFO` | Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |

### Execution Modes

| Mode | Strategy | Behavior |
|---|---|---|
| `block` | `MobileNetV2Strategy` | Fuses compatible layers (expand → DW → project) into multi-layer blocks sent as a single round-trip to each worker. Reduces communication overhead. |
| `layer` | `SingleLayerStrategy` | Every layer is executed individually. Useful for debugging, profiling per-layer behavior, or comparing against block mode. |

```bash
# Fused block execution (default, faster)
python main.py --workers 4 --mode block

# Per-layer execution (debug / profiling)
python main.py --workers 4 --mode layer --log-level DEBUG
```

## Protocol

The coordinator and workers communicate over TCP using a custom binary protocol (little-endian).

### Message Flow

```
Worker                          Coordinator
  │                                  │
  │──── REGISTER (clock_mhz) ──────►│
  │◄─── REGISTER_ACK (id) ─────────│
  │                                  │
  │◄─── TASK (layer params + data) ─│
  │──── RESULT (timing + data) ────►│
  │          ...                     │
  │◄─── SHUTDOWN ───────────────────│
  │                                  │
```

### Message Types

| Type | Direction | Description |
|---|---|---|
| `REGISTER` (0x01) | Worker → Coordinator | Worker announces itself with clock speed |
| `REGISTER_ACK` (0x02) | Coordinator → Worker | Coordinator assigns worker ID |
| `TASK` (0x03) | Coordinator → Worker | Layer parameters + input feature map slice |
| `RESULT` (0x04) | Worker → Coordinator | Compute/compress timing + output data |
| `ERROR` (0x05) | Worker → Coordinator | Error code + description |
| `HEARTBEAT` (0x06) | Worker → Coordinator | Keep-alive (reserved) |
| `SHUTDOWN` (0x07) | Coordinator → Worker | Graceful shutdown signal |

Every message starts with a 16-byte header:

```
┌─────────┬──────┬───────────┬─────────────┬──────────┐
│  Magic  │ Type │ Worker ID │ Payload Len │ Reserved │
│ 4 bytes │  1B  │    1B     │   4 bytes   │  6 bytes │
└─────────┴──────┴───────────┴─────────────┴──────────┘
```

## Model Configuration

The model is defined in `src/model_config.json`. Each layer entry contains:

```json
{
  "layer_config": {
    "name": "blk3_expand",
    "type": 3,
    "in_channels": 24,
    "out_channels": 144,
    "kernel_size": 1,
    "stride": 1,
    "padding": 0,
    "groups": 1,
    "residual_add_to": "blk3_res",
    "residual_connect_from": null
  },
  "quant_params": {
    "s_in": 0.037,
    "z_in": 57,
    "s_w": [0.001, ...],
    "z_w": [0, ...],
    "s_out": 0.02,
    "z_out": 120,
    "m": [0.05, ...],
    "s_residual_out": null,
    "z_residual_out": null
  }
}
```

Layer types: `CONV` (1), `DEPTHWISE` (2), `POINTWISE` (3), `FC` (4).

## Extending to New Models

To add support for a new model architecture:

1. **Create a new strategy** in `src/model/block_strategy.py`:

```python
class ResNetStrategy(BlockGroupingStrategy):
    """Recognise ResNet bottleneck patterns: conv1x1 → conv3x3 → conv1x1."""

    def group(self, layers, qps):
        # Your grouping logic here
        ...
```

2. **Register it** in the `STRATEGIES` dictionary:

```python
STRATEGIES = {
    'mobilenetv2': MobileNetV2Strategy,
    'single_layer': SingleLayerStrategy,
    'resnet': ResNetStrategy,  # ← add here
}
```

3. **Map a CLI mode** (optional) in `_MODE_TO_STRATEGY`:

```python
_MODE_TO_STRATEGY = {
    'block': 'mobilenetv2',
    'layer': 'single_layer',
    'resnet_block': 'resnet',  # ← add here
}
```

4. **Prepare the model config JSON** with your model's layers and quantization parameters.

No changes to the inference engine, distributor, or coordinator are needed — the strategy pattern handles all architecture-specific logic.

## Testing

Run the full test suite:

```bash
python -m unittest tests.test_coordinator_core -v
```

The tests cover:
- **Model loader** — JSON parsing, input quantization
- **Block strategies** — MobileNetV2 pattern detection, single-layer fallback, strategy registry
- **Inference engine** — GAP application before FC, layer routing (conv vs FC)
- **Task distributor** — Row splitting, result collection and assembly
- **Statistics** — Record lifecycle, reset behavior
- **Coordinator** — Sub-system initialization smoke test

## Logging

All logs are written to `coordinator.log` in the working directory. Use `--log-level DEBUG` to capture detailed per-worker timing and feature map shapes:

```bash
python main.py --workers 4 --log-level DEBUG
```

Example log output:

```
[2026-02-25 10:30:01] src.coordinator - INFO - [Coordinator]: Started on 192.168.1.10:54321
[2026-02-25 10:30:05] src.coordinator - INFO - [Coordinator]: Worker 0 registered with clock 600 MHz
[2026-02-25 10:30:06] src.coordinator - INFO - [Coordinator]: Grouped 53 layers into 20 blocks (mode=block)
[2026-02-25 10:30:12] src.inference.engine - INFO - [Engine]: Inference completed in 6.1234 seconds
[2026-02-25 10:30:12] src.stats - INFO - Layer     0 [    CONV] init_conv: total=312.45ms  compute=280.12ms
[2026-02-25 10:30:12] src.stats - INFO - Layer   1-3 [   BLOCK] blk0_dw+blk0_project: total=198.32ms ...
```

## License

This project is part of a research prototype for distributed inference on MCU clusters.