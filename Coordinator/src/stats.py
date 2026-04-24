"""Inference performance statistics tracking."""

import logging
import time
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


class InferenceStats:
    """Collects per-layer / per-block timing and byte-volume statistics."""

    def __init__(self) -> None:
        self.records: list[dict] = []
        self._current: dict = {}
        # cumulative byte counters across the whole inference
        self.total_send_bytes_actual: int = 0
        self.total_send_bytes_full: int = 0
        self.total_recv_bytes_actual: int = 0
        self.total_recv_bytes_full: int = 0

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
            "send_bytes_actual": 0,
            "send_bytes_full": 0,
            "recv_bytes_actual": 0,
            "recv_bytes_full": 0,
            "workers": {},
        }

    @property
    def current(self) -> dict:
        """Direct access for distributor to attach per-worker entries."""
        return self._current

    def record_worker_send(self, worker_id: int, send_time_ms: float,
                           send_bytes_actual: int = 0, send_bytes_full: int = 0) -> None:
        self._current["workers"][worker_id] = {
            "send_time_ms": send_time_ms,
            "recv_time_ms": 0.0,
            "mcu_compute_ms": 0.0,
            "mcu_compress_ms": 0.0,
            "send_bytes_actual": send_bytes_actual,
            "send_bytes_full": send_bytes_full,
            "recv_bytes_actual": 0,
            "recv_bytes_full": 0,
        }
        self._current["send_bytes_actual"] += send_bytes_actual
        self._current["send_bytes_full"] += send_bytes_full
        self.total_send_bytes_actual += send_bytes_actual
        self.total_send_bytes_full += send_bytes_full

    def record_worker_result(self, worker_id: int, compute_ms: float, compress_ms: float, recv_time_ms: float,
                             recv_bytes_actual: int = 0, recv_bytes_full: int = 0) -> None:
        if worker_id in self._current["workers"]:
            ws = self._current["workers"][worker_id]
            ws["mcu_compute_ms"] = compute_ms
            ws["mcu_compress_ms"] = compress_ms
            ws["recv_time_ms"] = recv_time_ms
            ws["recv_bytes_actual"] = recv_bytes_actual
            ws["recv_bytes_full"] = recv_bytes_full
        self._current["recv_bytes_actual"] += recv_bytes_actual
        self._current["recv_bytes_full"] += recv_bytes_full
        self.total_recv_bytes_actual += recv_bytes_actual
        self.total_recv_bytes_full += recv_bytes_full

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

    @staticmethod
    def _fmt_bytes(n: int) -> str:
        if n >= 1024 * 1024:
            return f"{n / (1024 * 1024):.2f}MB"
        if n >= 1024:
            return f"{n / 1024:.2f}KB"
        return f"{n}B"

    @staticmethod
    def _saving_pct(actual: int, full: int) -> str:
        if full <= 0:
            return "  n/a"
        return f"{(1 - actual / full) * 100:5.1f}%"

    def print_summary(self) -> None:
        logger.info("Inference execution stats:")
        for s in self.records:
            idx_str = str(s["layer_idx"])
            sa = s.get("send_bytes_actual", 0)
            sf = s.get("send_bytes_full", 0)
            ra = s.get("recv_bytes_actual", 0)
            rf = s.get("recv_bytes_full", 0)
            logger.info(
                f"Layer {idx_str:>5} [{s['layer_type']:>8}] {s['layer_name']}: "
                f"total={s['total_time_ms']:7.2f}ms  "
                f"compute={s.get('avg_compute_ms', 0):6.2f}ms  "
                f"compress={s.get('avg_compress_ms', 0):5.2f}ms  "
                f"send={self._fmt_bytes(sa):>9}/{self._fmt_bytes(sf):>9} (save {self._saving_pct(sa, sf)})  "
                f"recv={self._fmt_bytes(ra):>9}/{self._fmt_bytes(rf):>9} (save {self._saving_pct(ra, rf)})"
            )

        logger.info(
            f"Total send: actual={self._fmt_bytes(self.total_send_bytes_actual)}  "
            f"full-equiv={self._fmt_bytes(self.total_send_bytes_full)}  "
            f"saved={self._saving_pct(self.total_send_bytes_actual, self.total_send_bytes_full)}"
        )
        logger.info(
            f"Total recv: actual={self._fmt_bytes(self.total_recv_bytes_actual)}  "
            f"full-equiv={self._fmt_bytes(self.total_recv_bytes_full)}  "
            f"saved={self._saving_pct(self.total_recv_bytes_actual, self.total_recv_bytes_full)}"
        )

    def reset(self) -> None:
        self.records.clear()
        self._current = {}
        self.total_send_bytes_actual = 0
        self.total_send_bytes_full = 0
        self.total_recv_bytes_actual = 0
        self.total_recv_bytes_full = 0
