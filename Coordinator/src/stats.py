"""Inference performance statistics tracking."""

import logging
import time
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


class InferenceStats:
    """Collects per-layer / per-block timing statistics during inference."""

    def __init__(self) -> None:
        self.records: list[dict] = []
        self._current: dict = {}

    # ── Layer / block lifecycle ──────────────────────────────────────

    def begin_layer(
        self,
        layer_idx: int | str,
        layer_name: str,
        layer_type: str,
    ) -> None:
        """Call once before executing a layer or block."""
        self._current = {
            "layer_idx": layer_idx,
            "layer_name": layer_name,
            "layer_type": layer_type,
            "total_time_ms": 0.0,
            "avg_compute_ms": 0.0,
            "avg_compress_ms": 0.0,
            "workers": {},
        }

    @property
    def current(self) -> dict:
        """Direct access for distributor to attach per-worker entries."""
        return self._current

    def record_worker_send(self, worker_id: int, send_time_ms: float) -> None:
        self._current["workers"][worker_id] = {
            "send_time_ms": send_time_ms,
            "recv_time_ms": 0.0,
            "mcu_compute_ms": 0.0,
            "mcu_compress_ms": 0.0,
        }

    def record_worker_result(self, worker_id: int, compute_ms: float, compress_ms: float, recv_time_ms: float) -> None:
        if worker_id in self._current["workers"]:
            ws = self._current["workers"][worker_id]
            ws["mcu_compute_ms"] = compute_ms
            ws["mcu_compress_ms"] = compress_ms
            ws["recv_time_ms"] = recv_time_ms

    def end_layer(self, total_time_ms: float) -> None:
        """Finalise the current layer record and append to history."""
        self._current["total_time_ms"] = total_time_ms
        worker_stats = list(self._current["workers"].values())
        if worker_stats:
            self._current["avg_compute_ms"] = float(
                np.mean([ws["mcu_compute_ms"] for ws in worker_stats])
            )
            self._current["avg_compress_ms"] = float(
                np.mean([ws["mcu_compress_ms"] for ws in worker_stats])
            )
        self.records.append(self._current)
        self._current = {}

    # ── Reporting ────────────────────────────────────────────────────

    def print_summary(self) -> None:
        logger.info("Inference execution stats:")
        for s in self.records:
            idx_str = str(s["layer_idx"])
            logger.info(
                f"Layer {idx_str:>5} [{s['layer_type']:>8}] {s['layer_name']}: "
                f"total={s['total_time_ms']:.2f}ms  "
                f"compute={s.get('avg_compute_ms', 0):.2f}ms  "
                f"compress={s.get('avg_compress_ms', 0):.2f}ms"
            )

    def reset(self) -> None:
        self.records.clear()
        self._current = {}
