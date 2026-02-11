import asyncio
import logging
import time

from .protocol import *
from .work_manager import *
from .task_queue import TaskQueue

logging.basicConfig(filename='./coordinator.log', 
                    filemode='w',
                    level=logging.DEBUG, 
                    format='[%(asctime)s] %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class Coordinator:
    def __init__(self, host: str = '192, 168, 1, 10', port: int = 54321):
        self.host = host
        self.port = port
        self.worker_manager = WorkerManager()
        self.task_queue = TaskQueue()
        self.running = False

    async def start(self):
        self.running = True

        server = await asyncio.start_server(self.on_client_connected, self.host, self.port)
        logger.info(f"Coordinator started on {self.host}:{self.port}")

        tasks = [
            asyncio.create_task(server.serve_forever()), # start to listen
            asyncio.create_task(self.task_dispatcher()), # start to dispatch tasks
            # asyncio.create_task(self.worker_manager)
            asyncio.create_task(self.stats_printer()), # start to print stats
        ]
        try:
            await asyncio.gather(*tasks)
        except KeyboardInterrupt:
            logger.info("Shutting down coordinator...")
            self.running = False
            server.close()
            await server.wait_closed()
            logger.info("Coordinator stopped.")

    async def on_client_connected(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        worker = self.worker_manager.add_worker(reader, writer) # worker needs contains some info
        try:
            # once connected, worker will send a registration message
            # parse it and send corresponding ACK
            result: tuple[MessageHeader, bytes] = await self.worker_manager.receive_message(worker, timeout=10)
            if not result:
                logger.error("Failed to receive registration message from worker")
                return
            
            # check header and payload
            header, payload = result
            if header.type != MessageType.REGISTER:
                logger.error("Expected REGISTER message, got {header.type}")
                return
            reg_msg = RegisterMessage.unpack(payload)
            worker.clock_mhz = reg_msg.clock_mhz
            logger.info(f"Worker registered with clock {worker.clock_mhz} MHz")

            # send ACK
            ack_msg = RegisterAckMessage(status=0, assigned_id=worker.worker_id)
            await self.worker_manager.send_message(worker, MessageType.REGISTER_ACK, ack_msg.pack())

            # everything goes fine, worker -> idle, waiting for task assignment
            # go to event loop, waiting for messages from worker, which can be either RESULT or ERROR
            worker.state = WorkerState.IDLE
            await self.event_loop(worker)
            
        except Exception as e:
            logger.error(f"Error handling for worker {worker.worker_id}: {e}")
        finally:
            logger.info(f"Worker disconnected: {worker}")
            self.worker_manager.remove_worker(worker)
        
    async def event_loop(self, worker: WorkerInfo):
        while self.running and worker.state != WorkerState.DISCONNECTED:
            result: tuple[MessageHeader, bytes] = await self.worker_manager.receive_message(worker, timeout=1)
            if not result:
                continue
            
            header, payload = result
            
            if header.type == MessageType.RESULT:
                await self.handle_result(worker, payload)
            elif header.type == MessageType.ERROR:
                await self.handle_error(worker, payload)
            else:
                logger.warning(f"Unexpected message type {header.type} from worker {worker.worker_id}")

    async def handle_result(self, worker: WorkerInfo, payload: bytes):
        # parse the result and update task status
        logger.debug(f"Received result from worker {worker.worker_id}")
        pass

    async def handle_error(self, worker: WorkerInfo, payload: bytes):
        # parse the error message and log it
        logger.error(f"Received error from worker {worker.worker_id}")
        pass


    async def task_dispatcher(self):
        pass

    async def stats_printer(self):
        pass