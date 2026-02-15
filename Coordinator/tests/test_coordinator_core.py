import asyncio
import json
import struct
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import numpy as np

from src.coordniator import Coordinator, LayerConfig, QuantParams
from src.protocol import LayerType, MessageType, MessageHeader, ResultMessage
from src.work_manager import WorkerState


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


class TestCoordinatorCore(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.coordinator = Coordinator(host="127.0.0.1", port=54321)
        self.coordinator.worker_manager.workers = {
            0: _make_worker(0),
            1: _make_worker(1),
        }

    async def test_run_layer_fc_applies_gap_and_routes_fc(self):
        c = self.coordinator
        c.feature_map = np.random.randint(0, 255, size=(4, 2, 2), dtype=np.uint8)

        layer = LayerConfig(
            name="fc",
            type=LayerType.FC,
            layer_idx=0,
            in_channels=4,
            out_channels=10,
        )
        qp = QuantParams(
            s_in=0.1, z_in=128,
            s_w=np.array([0.1], dtype=np.float32),
            z_w=np.array([0], dtype=np.int32),
            s_out=0.2, z_out=120,
            m=np.array([0.05], dtype=np.float32),
        )

        c._distribute_fc = AsyncMock(return_value=None)
        c._distribute_conv = AsyncMock(return_value=None)

        await c._run_layer(layer, qp)

        self.assertEqual(c.feature_map.ndim, 1, "FC dimension should be 1D after GAP")
        self.assertEqual(c.feature_map.shape[0], 4)
        c._distribute_fc.assert_awaited_once()
        c._distribute_conv.assert_not_called()

    async def test_distribute_conv_dispatches_rows_when_padding_positive(self):
        c = self.coordinator
        c.feature_map = np.random.randint(0, 255, size=(3, 4, 4), dtype=np.uint8)

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
        qp = QuantParams(
            s_in=0.1, z_in=128,
            s_w=np.array([0.1], dtype=np.float32),
            z_w=np.array([0], dtype=np.int32),
            s_out=0.2, z_out=120,
            m=np.array([0.05], dtype=np.float32),
        )

        c._send_task_to_worker = AsyncMock(return_value=True)
        c._collect_results = AsyncMock(
            return_value=np.zeros((8, 4, 4), dtype=np.uint8)
        )

        await c._distribute_conv(layer, qp)

        # expect at least one task sent to worker
        self.assertGreater(
            c._send_task_to_worker.await_count, 0,
            "padding>0 时也应分发任务到 worker"
        )
        c._collect_results.assert_awaited_once()

        tasks_arg = c._collect_results.await_args.args[0]
        covered_rows = set()
        for _, start, end, _ in tasks_arg:
            covered_rows.update(range(start, end))

        # H_out = 4
        self.assertEqual(covered_rows, {0, 1, 2, 3}, "分片应覆盖全部输出行")

    async def test_receive_worker_result_writes_conv_slice(self):
        c = self.coordinator
        worker = self.coordinator.worker_manager.workers[0]

        output = np.zeros((2, 4, 3), dtype=np.uint8)  # C=2,H=4,W=3
        start_idx, end_idx = 1, 3  # H_slice=2
        patch = np.arange(2 * 2 * 3, dtype=np.uint8).reshape(2, 2, 3)

        payload = struct.pack(ResultMessage.FORMAT, 1234, patch.size)
        header = MessageHeader(
            type=MessageType.RESULT,
            worker_id=worker.worker_id,
            payload_len=len(payload),
        )

        c.worker_manager.receive_message = AsyncMock(return_value=(header, payload))
        worker.reader.readexactly = AsyncMock(return_value=patch.tobytes())
        c.worker_manager.mark_worker_idle = MagicMock()

        await c._receive_worker_result(
            worker=worker,
            start_idx=start_idx,
            end_idx=end_idx,
            output=output,
        )

        np.testing.assert_array_equal(output[:, 1:3, :], patch)
        c.worker_manager.mark_worker_idle.assert_called_once()

    def test_parse_layer_configs_from_json(self):
        c = self.coordinator

        fake_cfg = {
            "layers": [
                {
                    "layer_config": {
                        "name": "conv0",
                        "type": "CONV",
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
                    },
                }
            ]
        }

        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "model_config.json"
            p.write_text(json.dumps(fake_cfg), encoding="utf-8")

            c._parse_layer_configs(str(p))

        self.assertEqual(len(c.layer_config_list), 1)
        self.assertEqual(len(c.quant_params_list), 1)
        self.assertEqual(c.layer_config_list[0].name, "conv0")
        self.assertEqual(c.layer_config_list[0].layer_idx, 0)
        self.assertEqual(c.quant_params_list[0].z_in, 128)


if __name__ == "__main__":
    unittest.main()