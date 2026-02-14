import asyncio
import logging
from enum import Enum
from dataclasses import dataclass
from .protocol import *
# from .task_queue import *

logger = logging.getLogger(__name__)

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
        self.workers: dict[int, WorkerInfo] = {}
        self.next_worker_id = 0

        # idle worker queue
        self.idle_queue: asyncio.Queue[WorkerInfo] = asyncio.Queue()

    def add_worker(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> WorkerInfo:
        logger.info(f"[WorkerManager]: Adding new worker from {writer.get_extra_info('peername')}")
        worker_id = self.next_worker_id
        self.next_worker_id += 1
        worker = WorkerInfo(worker_id=worker_id, clock_mhz=0, reader=reader, writer=writer, state=WorkerState.CONNECTED)
        self.workers[worker_id] = worker
        return worker

    def remove_worker(self, worker: WorkerInfo):
        if worker.worker_id in self.workers:
            try:
                worker.writer.close()
                asyncio.create_task(worker.writer.wait_closed())
            except Exception as e:
                logger.error(f"[WorkerManager]: Error closing connection for worker {worker.worker_id}: {e}")
            
            del self.workers[worker.worker_id]

    async def send_message(self, worker: WorkerInfo, msg_type: MessageType, payload: bytes):
        try:
            header = MessageHeader(type=msg_type, worker_id=worker.worker_id, payload_len=len(payload))
            worker.writer.write(header.pack())
            if payload:
                worker.writer.write(payload)
            await worker.writer.drain()
            
            return True
        
        except Exception as e:
            logger.error(f"[WorkerManager]: Error sending message to worker {worker.worker_id}: {e}")
            worker.state = WorkerState.DISCONNECTED
            return False
    
    async def receive_message(self, worker: WorkerInfo, timeout=None) -> tuple[MessageHeader, bytes]:
        try:
            logger.debug(f"[WorkerManager]: Waiting for message from worker {worker.worker_id} with timeout {timeout}")
            header_data = await asyncio.wait_for(worker.reader.readexactly(MessageHeader.SIZE), timeout=timeout)
            logger.debug(f"Received header data: {header_data.hex()}")
            header = MessageHeader.unpack(header_data)

            payload = b''    
            if header.payload_len > 0:
                payload = await asyncio.wait_for(worker.reader.readexactly(header.payload_len), timeout=timeout)
            
            logger.debug(f"[WorkerManager]: Received message of type {header.type} with payload length {header.payload_len} from worker {worker.worker_id}")
            return header, payload
        
        except asyncio.TimeoutError:
            logger.warning(f"[WorkerManager]: Timeout while waiting for message from worker {worker.worker_id}")
            return None
        except Exception as e:
            logger.error(f"[WorkerManager]: Error receiving message from worker {worker.worker_id}: {e}")
            worker.state = WorkerState.DISCONNECTED
            return None
        
    def mark_worker_idle(self, worker: WorkerInfo):
        if worker.state != WorkerState.IDLE:
            worker.state = WorkerState.IDLE
            try:
                self.idle_queue.put_nowait(worker)
            except asyncio.QueueFull:
                logger.warning(f"[WorkerManager]: Idle queue is full")
            
    async def heartbeat_monitor(self):
        pass

