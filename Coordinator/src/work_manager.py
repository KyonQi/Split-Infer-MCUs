import asyncio
from enum import Enum
from dataclasses import dataclass
from .protocol import *


class WorkerState(IntEnum):
    DISCONNECTED = 0
    CONNECTED = 1
    REGISTERED = 2
    IDLE = 3
    BUSY = 4


@dataclass
class WorkerInfo:
    worker_id: int
    clock_mhz: int
    reader: asyncio.StreamReader
    writer: asyncio.StreamWriter
    state: WorkerState = WorkerState.DISCONNECTED

class WorkerManager:
    def __init__(self):
        pass

    def add_worker(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> WorkerInfo:
        pass

    def remove_worker(self, worker: WorkerInfo):
        pass

    async def send_message(self, worker: WorkerInfo, msg_type: MessageType, payload: bytes):
        pass

    async def receive_message(self, worker: WorkerInfo, timeout=None) -> tuple[MessageHeader, bytes]:
        pass

