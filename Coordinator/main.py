import asyncio
import logging
from src.coordniator import Coordinator

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