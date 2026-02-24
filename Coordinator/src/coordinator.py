"""Coordinator — TCP server, worker registration, and inference orchestration.
``InferenceEngine`` (execution), ``TaskDistributor`` (communication), and
``BlockGroupingStrategy`` (layer grouping).
"""

from __future__ import annotations

import asyncio
import logging
import socket
from typing import Optional

import numpy as np

from .config import CoordinatorConfig
from .inference.distributor import TaskDistributor
from .inference.engine import InferenceEngine
from .model.block_strategy import get_strategy
from .model.loader import parse_layer_configs, quantize_input
# from .model.types import BlockConfig, LayerConfig, QuantParams
from .protocol import LayerType, MessageType, RegisterAckMessage, RegisterMessage
from .stats import InferenceStats
from .work_manager import WorkerManager, WorkerState

logger = logging.getLogger(__name__)


class Coordinator:
    """
    Central coordinator that manages workers and drives inference.

    Parameters
    ----------
    config:
        Runtime configuration (host, port, execution mode, model path).
    """

    def __init__(self, config: CoordinatorConfig) -> None:
        self.config = config
        self.running = False

        # Sub-systems
        self.worker_manager = WorkerManager()
        self.stats = InferenceStats()
        self.distributor = TaskDistributor(self.worker_manager, self.stats)
        self.engine = InferenceEngine(self.distributor, self.stats)

    # ------------------------------------------------------------------
    # TCP server lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        self.running = True
        server = await asyncio.start_server(
            self._on_client_connected, self.config.host, self.config.port
        )
        logger.info(f"[Coordinator]: Started on {self.config.host}:{self.config.port}")

        try:
            await server.serve_forever()
        except KeyboardInterrupt:
            logger.info("[Coordinator]: Shutting down…")
            self.running = False
            server.close()
            await server.wait_closed()
            logger.info("[Coordinator]: Stopped.")

    # ------------------------------------------------------------------
    # Worker registration
    # ------------------------------------------------------------------

    async def _on_client_connected(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        """Handle a new TCP connection: register the worker."""
        sock: socket.socket = writer.get_extra_info("socket")
        if sock is not None:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            logger.info("[Coordinator]: Set TCP_NODELAY on socket")

        logger.info(f"[Coordinator]: New worker connected from {writer.get_extra_info('peername')}")
        worker = self.worker_manager.add_worker(reader, writer)

        try:
            result = await self.worker_manager.receive_message(worker, timeout=2)
            if not result:
                logger.error("[Coordinator]: Failed to receive registration message")
                self.worker_manager.remove_worker(worker)
                return

            header, payload = result
            if header.type != MessageType.REGISTER:
                logger.error(f"[Coordinator]: Expected REGISTER, got {header.type}")
                self.worker_manager.remove_worker(worker)
                return

            reg_msg = RegisterMessage.unpack(payload)
            worker.worker_id = header.worker_id
            worker.clock_mhz = reg_msg.clock_mhz
            logger.info(
                f"[Coordinator]: Worker {worker.worker_id} registered "
                f"with clock {worker.clock_mhz} MHz"
            )

            ack_msg = RegisterAckMessage(status=0, assigned_id=worker.worker_id)
            await self.worker_manager.send_message(worker, MessageType.REGISTER_ACK, ack_msg.pack())

            self.worker_manager.mark_worker_idle(worker)

        except Exception as e:
            logger.error(f"[Coordinator]: Error during worker {worker.worker_id} registration: {e}")
            worker.state = WorkerState.DISCONNECTED
            self.worker_manager.remove_worker(worker)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    async def execute_inference(self, input_data: np.ndarray) -> np.ndarray:
        """Run a complete inference pass on *input_data*.

        1. Parse model config
        2. Quantize input
        3. Group layers using the configured strategy
        4. Delegate execution to InferenceEngine
        """
        layers, qps = parse_layer_configs(self.config.model_config_path)
        quantized = quantize_input(input_data, qps[0])

        strategy = get_strategy(self.config.execution_mode)
        blocks = strategy.group(layers, qps)
        logger.info(
            f"[Coordinator]: Grouped {len(layers)} layers into {len(blocks)} blocks "
            f"(mode={self.config.execution_mode})"
        )

        self.stats.reset()
        return await self.engine.execute(quantized, layers, qps, blocks)

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    async def shutdown_workers(self) -> None:
        """Gracefully shut down all connected workers."""
        await self.distributor.shutdown_workers()
