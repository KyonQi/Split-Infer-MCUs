import asyncio
import logging
from src.coordniator import Coordinator

logging.basicConfig(filename='./coordinator.log', 
                    filemode='w',
                    level=logging.DEBUG, 
                    format='[%(asctime)s] %(name)s - %(levelname)s: %(message)s')
# Set root logger level explicitly to ensure all child loggers inherit it
logging.getLogger().setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)

async def main():
    coord = Coordinator(host='192.168.1.10', port=54321)
    print("Coordinator is starting...\n")
    logger.info("Coordinator is starting...")
    await coord.start()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nCoordinator is shutting down...\n")
        logger.info("Coordinator is shutting down...")