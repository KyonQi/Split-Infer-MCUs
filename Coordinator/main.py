import asyncio
import contextlib
import logging
import argparse
import torch
from torchvision import transforms
import numpy as np
from pathlib import Path
from PIL import Image
from src.coordniator import Coordinator

logging.basicConfig(filename='./coordinator.log', 
                    filemode='w',
                    level=logging.DEBUG, 
                    format='[%(asctime)s] %(name)s - %(levelname)s - [%(filename)s:%(lineno)d]: %(message)s')
logging.getLogger().setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)

def prepocess_image(image_path: str) -> np.ndarray:
    prepocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img = Image.open(image_path).convert("RGB")
    input_tensor: torch.Tensor = prepocess(img)
    return input_tensor.numpy()

async def wait_for_workers(coord: Coordinator, num_workers: int):
    while len(coord.worker_manager.workers.values()) < num_workers:
        await asyncio.sleep(1)
    logger.info(f"All {len(coord.worker_manager.workers.values())} workers have connected.")

async def main(workers: int):
    coord = Coordinator(host='192.168.1.10', port=54321)
    print("Coordinator is starting...\n")
    logger.info("Coordinator is starting...")
    server_task = asyncio.create_task(coord.start()) # start will block until the server is closed so we run it in a separate task
    await wait_for_workers(coord, workers) # block until all workes have connected
    try:
        # Load and prepare input data (example)
        input_image_path = Path("./data/panda.jpg")
        input_image = prepocess_image(str(input_image_path))
        # input_image = np.random.rand(3, 224, 224).astype(np.float32)
        output = await coord.execute_inference(input_image)
        logger.debug(f"Inference output: {output}")
    finally:
        # send shutdown message to all workers so they can clean up and exit gracefully
        await coord.shutdown_workers()
        coord.running = False
        server_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await server_task
        logger.info("Coordinator has shut down.")        


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Coordinator for distributed DNN inference")
    parser.add_argument('--workers', type=int, default=2, help='Number of workers')
    args = parser.parse_args()
    
    try:
        asyncio.run(main(args.workers))
    except KeyboardInterrupt:
        print("\nCoordinator is shutting down...\n")
        logger.info("Coordinator is shutting down...")