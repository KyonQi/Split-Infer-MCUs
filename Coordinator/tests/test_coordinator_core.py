"""Tests covering the refactored coordinator sub-modules.

Each test targets a specific module:
  - model/loader.py     → parse_layer_configs, quantize_input
  - model/block_strategy → MobileNetV2Strategy, SingleLayerStrategy
  - inference/engine.py  → InferenceEngine._run_layer (GAP + routing)
  - inference/distributor → TaskDistributor.distribute_conv, _receive_worker_result
"""

import asyncio
import json
import struct
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np

from src.config import CoordinatorConfig
from src.coordinator import Coordinator
from src.inference.distributor import TaskDistributor
from src.inference.engine import InferenceEngine
from src.model.block_strategy import (
    MobileNetV2Strategy,
    SingleLayerStrategy,
    get_strategy,
)
from src.model.loader import parse_layer_configs, quantize_input
from src.model.types import BlockConfig, LayerConfig, QuantParams
from src.protocol import LayerType, MessageHeader, MessageType, ResultMessage
from src.stats import InferenceStats
from src.work_manager import WorkerManager, WorkerState


# ── Helpers ──────────────────────────────────────────────────────────


def _make_worker(worker_id: int):
    reader = AsyncMock()
    writer = MagicMock()
    return SimpleNamespace(
        worker_id=worker_id,
        clock_mhz=600,
        reader=reader,
        writer=writer,
        state=WorkerState.IDLE,
    )


def _make_qp(**overrides) -> QuantParams:
    defaults = dict(
        s_in=0.1,
        z_in=128,
        s_w=np.array([0.1], dtype=np.float32),
        z_w=np.array([0], dtype=np.int32),
        s_out=0.2,
        z_out=120,
        m=np.array([0.05], dtype=np.float32),
    )
    defaults.update(overrides)
    return QuantParams(**defaults)


# ── model/loader tests ───────────────────────────────────────────────


class TestModelLoader(unittest.TestCase):
    def test_parse_layer_configs_from_json(self):
        fake_cfg = {
            "layers": [
                {
                    "layer_config": {
                        "name": "conv0",
                        "type": 1,  # CONV
                        "in_channels": 3,
                        "out_channels": 8,
                        "kernel_size": 3,
                        "stride": 1,
                        "padding": 1,
                        "groups": 1,
                        "residual_add_to": None,
                        "residual_connect_from": None,
                    },
                    "quant_params": {
                        "s_in": 0.1,
                        "z_in": 128,
                        "s_w": [0.1] * 8,
                        "z_w": [0] * 8,
                        "s_out": 0.2,
                        "z_out": 120,
                        "m": [0.05] * 8,
                        "s_residual_out": None,
                        "z_residual_out": None,
                    },
                }
            ]
        }

        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "model_config.json"
            p.write_text(json.dumps(fake_cfg), encoding="utf-8")

            layers, qps = parse_layer_configs(str(p))

        self.assertEqual(len(layers), 1)
        self.assertEqual(len(qps), 1)
        self.assertEqual(layers[0].name, "conv0")
        self.assertEqual(layers[0].layer_idx, 0)
        self.assertEqual(qps[0].z_in, 128)

    def test_quantize_input(self):
        data = np.array([0.0, 0.1, 0.2], dtype=np.float32)
        qp = _make_qp(s_in=0.1, z_in=10)
        result = quantize_input(data, qp)
        self.assertEqual(result.dtype, np.uint8)
        # 0.0 / 0.1 + 10 = 10
        self.assertEqual(result[0], 10)


# ── model/block_strategy tests ───────────────────────────────────────


class TestBlockStrategy(unittest.TestCase):
    def _make_layers(self, specs):
        """Build LayerConfig list from (name, type, kernel_size) triples."""
        layers = []
        for i, (name, lt, ks) in enumerate(specs):
            layers.append(
                LayerConfig(
                    name=name,
                    type=lt,
                    layer_idx=i,
                    in_channels=16,
                    out_channels=16,
                    kernel_size=ks,
                )
            )
        return layers

    def test_mobilenetv2_three_layer_block(self):
        specs = [
            ("expand", LayerType.POINTWISE, 1),
            ("dw", LayerType.DEPTHWISE, 3),
            ("project", LayerType.POINTWISE, 1),
        ]
        layers = self._make_layers(specs)
        qps = [_make_qp() for _ in layers]

        strategy = MobileNetV2Strategy()
        blocks = strategy.group(layers, qps)

        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0].start_idx, 0)
        self.assertEqual(blocks[0].end_idx, 2)
        self.assertEqual(len(blocks[0].layers), 3)

    def test_mobilenetv2_two_layer_block(self):
        specs = [
            ("dw", LayerType.DEPTHWISE, 3),
            ("project", LayerType.POINTWISE, 1),
        ]
        layers = self._make_layers(specs)
        qps = [_make_qp() for _ in layers]

        blocks = MobileNetV2Strategy().group(layers, qps)
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0].end_idx, 1)

    def test_single_layer_strategy(self):
        specs = [
            ("expand", LayerType.POINTWISE, 1),
            ("dw", LayerType.DEPTHWISE, 3),
            ("project", LayerType.POINTWISE, 1),
        ]
        layers = self._make_layers(specs)
        qps = [_make_qp() for _ in layers]

        blocks = SingleLayerStrategy().group(layers, qps)
        self.assertEqual(len(blocks), 3)
        for blk in blocks:
            self.assertEqual(blk.start_idx, blk.end_idx)

    def test_get_strategy_with_mode(self):
        s = get_strategy('block')
        self.assertIsInstance(s, MobileNetV2Strategy)
        s = get_strategy('layer')
        self.assertIsInstance(s, SingleLayerStrategy)

    def test_get_strategy_unknown_raises(self):
        with self.assertRaises(ValueError):
            get_strategy('unknown_model')


# ── inference/engine tests ───────────────────────────────────────────


class TestInferenceEngine(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.stats = InferenceStats()
        self.distributor = MagicMock(spec=TaskDistributor)
        self.engine = InferenceEngine(self.distributor, self.stats)

    async def test_run_layer_fc_applies_gap_and_routes_fc(self):
        engine = self.engine
        engine.feature_map = np.random.randint(0, 255, size=(4, 2, 2), dtype=np.uint8)
        engine.current_layer_idx = 0

        layer = LayerConfig(
            name="fc",
            type=LayerType.FC,
            layer_idx=0,
            in_channels=4,
            out_channels=10,
        )
        qp = _make_qp()

        # Mock the distributor to return a dummy result
        self.distributor.distribute_fc = AsyncMock(
            return_value=np.zeros(10, dtype=np.uint8)
        )
        self.distributor.distribute_conv = AsyncMock()

        await engine._run_layer(layer, qp)

        # After GAP, feature_map should have been 1D (4,) before distribute_fc was called
        self.distributor.distribute_fc.assert_awaited_once()
        self.distributor.distribute_conv.assert_not_called()
        # The final feature_map should be what distribute_fc returned
        self.assertEqual(engine.feature_map.shape, (10,))

    async def test_run_layer_conv_routes_conv(self):
        engine = self.engine
        engine.feature_map = np.random.randint(0, 255, size=(3, 4, 4), dtype=np.uint8)
        engine.current_layer_idx = 0

        layer = LayerConfig(
            name="conv",
            type=LayerType.CONV,
            layer_idx=0,
            in_channels=3,
            out_channels=8,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        qp = _make_qp()

        self.distributor.distribute_conv = AsyncMock(
            return_value=np.zeros((8, 4, 4), dtype=np.uint8)
        )

        await engine._run_layer(layer, qp)

        self.distributor.distribute_conv.assert_awaited_once()


# ── inference/distributor tests ──────────────────────────────────────


class TestTaskDistributor(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.stats = InferenceStats()
        self.wm = WorkerManager()
        self.wm.workers = {
            0: _make_worker(0),
            1: _make_worker(1),
        }
        self.dist = TaskDistributor(self.wm, self.stats)

    async def test_distribute_conv_dispatches_rows(self):
        feature_map = np.random.randint(0, 255, size=(3, 4, 4), dtype=np.uint8)

        layer = LayerConfig(
            name="conv",
            type=LayerType.CONV,
            layer_idx=0,
            in_channels=3,
            out_channels=8,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=1,
        )
        qp = _make_qp()

        self.dist._send_task_to_worker = AsyncMock(return_value=True)
        self.dist._collect_results = AsyncMock(
            return_value=np.zeros((8, 4, 4), dtype=np.uint8)
        )

        await self.dist.distribute_conv(feature_map, layer, qp, current_layer_idx=0)

        self.assertGreater(
            self.dist._send_task_to_worker.await_count,
            0,
            "Should dispatch tasks to workers",
        )
        self.dist._collect_results.assert_awaited_once()

        tasks_arg = self.dist._collect_results.await_args.args[0]
        covered_rows = set()
        for _, start, end, _ in tasks_arg:
            covered_rows.update(range(start, end))
        self.assertEqual(covered_rows, {0, 1, 2, 3}, "Slices should cover all output rows")

    async def test_receive_worker_result_writes_conv_slice(self):
        self.stats.begin_layer(0, "conv", "CONV")

        worker = self.wm.workers[0]
        output = np.zeros((2, 4, 3), dtype=np.uint8)
        start_idx, end_idx = 1, 3
        patch = np.arange(2 * 2 * 3, dtype=np.uint8).reshape(2, 2, 3)

        payload = struct.pack(ResultMessage.FORMAT, 1234, 0, patch.size)
        header = MessageHeader(
            type=MessageType.RESULT,
            worker_id=worker.worker_id,
            payload_len=len(payload),
        )

        self.wm.receive_message = AsyncMock(return_value=(header, payload))
        worker.reader.readexactly = AsyncMock(return_value=patch.tobytes())
        self.wm.mark_worker_idle = MagicMock()

        await self.dist._receive_worker_result(
            worker=worker,
            start_idx=start_idx,
            end_idx=end_idx,
            output=output,
            current_layer_idx=0,
        )

        np.testing.assert_array_equal(output[:, 1:3, :], patch)
        self.wm.mark_worker_idle.assert_called_once()


# ── stats tests ──────────────────────────────────────────────────────


class TestInferenceStats(unittest.TestCase):
    def test_begin_end_layer_records(self):
        stats = InferenceStats()
        stats.begin_layer(0, "conv0", "CONV")
        stats.record_worker_send(0, 1.5)
        stats.record_worker_result(0, compute_ms=10.0, compress_ms=2.0, recv_time_ms=3.0)
        stats.end_layer(total_time_ms=20.0)

        self.assertEqual(len(stats.records), 1)
        rec = stats.records[0]
        self.assertEqual(rec["layer_idx"], 0)
        self.assertAlmostEqual(rec["total_time_ms"], 20.0)
        self.assertAlmostEqual(rec["avg_compute_ms"], 10.0)

    def test_reset_clears_records(self):
        stats = InferenceStats()
        stats.begin_layer(0, "conv0", "CONV")
        stats.end_layer(5.0)
        self.assertEqual(len(stats.records), 1)
        stats.reset()
        self.assertEqual(len(stats.records), 0)


# ── Coordinator integration smoke test ───────────────────────────────


class TestCoordinatorInit(unittest.TestCase):
    def test_default_config_creates_subsystems(self):
        config = CoordinatorConfig(host="127.0.0.1", port=54321)
        coord = Coordinator(config)

        self.assertIsInstance(coord.worker_manager, WorkerManager)
        self.assertIsInstance(coord.stats, InferenceStats)
        self.assertIsInstance(coord.distributor, TaskDistributor)
        self.assertIsInstance(coord.engine, InferenceEngine)


if __name__ == "__main__":
    unittest.main()
