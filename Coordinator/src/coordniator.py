import asyncio
import logging
import sys
import time
import json
import numpy as np
from dataclasses import dataclass
from typing import Optional, Union
from .protocol import *
from .work_manager import *
# from .task_queue import *

logger = logging.getLogger(__name__)

@dataclass
class LayerConfig:
    """layer config for inference execution"""
    name: str
    type: LayerType
    layer_idx: int
    in_channels: int
    out_channels: int
    kernel_size: int = 1
    stride: int = 1
    padding: int = 0
    groups: int = 1
    in_h: int = 0
    in_w: int = 0
    residual_add_to: Optional[str] = None
    residual_connect_from: Optional[str] = None

@dataclass
class QuantParams:
    """ quantization parameters needs to be shared between coordinator and workers """
    s_in: float
    z_in: int
    s_w: Union[float, np.ndarray]
    z_w: Union[float, np.ndarray]
    s_out: float
    z_out: int
    m: Union[float, np.ndarray] #float # precomputing multiplier for requantization m = (s_in * s_w) / s_out
    s_residual_out: Optional[float] = None
    z_residual_out: Optional[int] = None        

class Coordinator:
    def __init__(self, host: str = '192, 168, 1, 10', port: int = 54321):
        self.host: str = host
        self.port: int = port
        self.running = False
        self.worker_manager = WorkerManager()
        
        # inference managements
        self.feature_map: Optional[np.ndarray] = None
        self.residual_buffers: dict[str, tuple[np.ndarray, float, int]] = {}
        self.current_layer_idx: int = 0
        self.layer_config_list: list[LayerConfig] = [] # get the real vale by parsing the json file later
        self.quant_params_list: list[QuantParams] = [] # get the real value from calibration later

        # stats
        self.stats: list[dict] = []
        self.current_layer_stats: dict = {}

    async def start(self):
        self.running = True

        server = await asyncio.start_server(self.on_client_connected, self.host, self.port)
        logger.info(f"[Coordinator]: Coordinator started on {self.host}:{self.port}")

        tasks = [
            asyncio.create_task(server.serve_forever()), # start to listen
            # asyncio.create_task(self.task_dispatcher()), # start to dispatch tasks
            asyncio.create_task(self.worker_manager.heartbeat_monitor()), # start to monitor worker heartbeat
            # asyncio.create_task(self.stats_printer()), # start to print stats
        ]
        try:
            await asyncio.gather(*tasks)
        except KeyboardInterrupt:
            logger.info("[Coordinator]: Shutting down coordinator...")
            self.running = False
            server.close()
            await server.wait_closed()
            logger.info("[Coordinator]: Coordinator stopped.")

    async def on_client_connected(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        """ connected callback """
        logger.info(f"[Coordinator]: New worker connected from {writer.get_extra_info('peername')}")
        worker = self.worker_manager.add_worker(reader, writer) # worker needs contains some info
        try:
            # once connected, worker will send a registration message
            # parse it and send corresponding ACK
            result: tuple[MessageHeader, bytes] = await self.worker_manager.receive_message(worker, timeout=2)
            if not result:
                logger.error("[Coordinator]: Failed to receive registration message from worker")
                self.worker_manager.remove_worker(worker)
                return
            
            # check header and payload
            header, payload = result
            if header.type != MessageType.REGISTER:
                logger.error(f"[Coordinator]: Expected REGISTER message, got {LayerType(header.type)}")
                self.worker_manager.remove_worker(worker)   
                return
            reg_msg = RegisterMessage.unpack(payload)
            worker.worker_id = header.worker_id # notice here we change to the real hardware assigned worker id after registration
            worker.clock_mhz = reg_msg.clock_mhz
            logger.info(f"[Coordinator]: Worker {worker.worker_id} registered with clock {worker.clock_mhz} MHz")

            # send ACK
            ack_msg = RegisterAckMessage(status=0, assigned_id=worker.worker_id)
            await self.worker_manager.send_message(worker, MessageType.REGISTER_ACK, ack_msg.pack())

            # TODO we need 3 steps handshake for better synchronization, but currently we just assume everything goes fine after registration

            # everything goes fine, worker -> idle, waiting for task assignment
            # go to event loop, waiting for messages from worker, which can be either RESULT or ERROR
            # worker.state = WorkerState.IDLE
            self.worker_manager.mark_worker_idle(worker)
            # await self.worker_event_loop(worker)
            
        except Exception as e:
            logger.error(f"[Coordinator]: Error handling for worker {worker.worker_id}: {e}")
            worker.state = WorkerState.DISCONNECTED
            self.worker_manager.remove_worker(worker)
        # finally:
        #     logger.info(f"[Coordinator]: Worker {worker.worker_id} disconnected")
        #     worker.state = WorkerState.DISCONNECTED
        #     self.worker_manager.remove_worker(worker)
    
    async def execute_inference(self, input_data: np.ndarray) -> np.ndarray:
        logger.info(f"[Coordinator]: Starting inference execution for input shape {input_data.shape}")

        self._parse_layer_configs() # parse the layer config and quant params from json file, and fill in the layer_config_list and quant_params_list
        self.feature_map = self._quantize_input(input_data, self.quant_params_list[0]) # quantize the input data to uint8, and fill in the feature_map
        self.residual_buffers.clear()
        
        start_time = time.time()
        for layer_idx, (layer, quant_params) in enumerate(zip(self.layer_config_list, self.quant_params_list)):
            self.current_layer_idx = layer_idx
            logger.debug(f"[Coordinator]: Executing layer {layer_idx} - {layer.name} ({LayerType(layer.type)})")
            
            # init current layer stats
            self.current_layer_stats = {
                "layer_idx": layer_idx,
                "layer_name": layer.name,
                "layer_type": LayerType(layer.type).name,
                "total_time_ms": 0.0,
                "avg_compute_ms": 0.0,
                "avg_comm_ms": 0.0,
                "workers": {}, #
            }

            layer_start = time.perf_counter()
            await self._run_layer(layer, quant_params)
            layer_time = time.perf_counter() - layer_start
                        
            logger.debug(f"[Coordinator]: Layer {layer_idx} completed in {layer_time:.4f} seconds, output shape {self.feature_map.shape}")
        
            # current layer stats
            self.current_layer_stats["total_time_ms"] = layer_time * 1000
            worker_stats = list(self.current_layer_stats["workers"].values())
            if worker_stats:
                self.current_layer_stats["avg_compute_ms"] = float(np.mean([
                    ws["mcu_compute_ms"] for ws in worker_stats
                ]))
                # self.current_layer_stats["avg_comm_ms"] = float(np.mean([
                #     ws["send_time_ms"] + ws["recv_time_ms"] for ws in worker_stats
                # ])) # it's not reasonable, since we're using asycnio, read/write only relates to the data buffer
            self.stats.append(self.current_layer_stats) # store the last layer stats
            logger.debug(
                f"[Coordinator]: Layer {layer_idx} done — "
                f"total={self.current_layer_stats['total_time_ms']:.2f}ms  "
                f"compute={self.current_layer_stats['avg_compute_ms']:.2f}ms  "
                # f"comm={self.current_layer_stats['avg_comm_ms']:.2f}ms"
            )

        total_time = time.time() - start_time
        logger.info(f"[Coordinator]: Inference execution completed in {total_time:.4f} seconds")
        self.print_stats()

        return self.feature_map

    async def _run_layer(self, layer: LayerConfig, quant_params: QuantParams):
        if layer.residual_add_to:
            self.residual_buffers[layer.residual_add_to] = (self.feature_map.copy(), quant_params.s_in, quant_params.z_in)
            logger.debug(f"[Coordinator]: Stored residual buffer for {layer.residual_add_to} with shape {self.feature_map.shape}")
        
        # before fc, we needs a global average pooling and flatten
        if layer.type == LayerType.FC and self.feature_map.ndim == 3:
            gap_output = np.mean(self.feature_map, axis=(1, 2))
            self.feature_map = np.round(gap_output).astype(np.uint8)
            logger.debug(f"[Coordinator]: Applied GAP for FC layer, new shape {self.feature_map.shape}")
            with np.printoptions(threshold=sys.maxsize, linewidth=150):
                logger.debug(f"[Coordinator]: Sample GAP output values:\n{self.feature_map}\n")

        if layer.type == LayerType.FC:
            await self._distribute_fc(layer, quant_params)
        else:
            # deal with both conv2d and depthwise
            await self._distribute_conv(layer, quant_params)
        
        # apply residual
        if layer.residual_connect_from:
            await self._apply_residual(layer.residual_connect_from)
    

    async def _distribute_conv(self, layer: LayerConfig, quant_params: QuantParams):
        """Split the feature map by rows"""
        C, H, W = self.feature_map.shape
        H_out = (H + 2 * layer.padding - layer.kernel_size) // layer.stride + 1
        W_out = (W + 2 * layer.padding - layer.kernel_size) // layer.stride + 1
        
        if layer.padding > 0:
            padded = np.pad(
                self.feature_map,
                ((0, 0), (layer.padding, layer.padding), (layer.padding, layer.padding)),
                mode='constant',
                constant_values=quant_params.z_in
            )
        else:
            padded = self.feature_map
        
        available_workers = list(self.worker_manager.workers.values())
        num_workers = len(available_workers) # TODO maybe get idle workers
        rows_per_worker = int(np.ceil(H_out / num_workers))
        tasks = []
        
        for i, worker in enumerate(available_workers):
            start_row = i * rows_per_worker
            end_row = min(start_row + rows_per_worker, H_out)

            if start_row >= H_out:
                continue

            in_start_y = start_row * layer.stride
            in_end_y = (end_row - 1) * layer.stride + layer.kernel_size
            input_patch = padded[:, in_start_y:in_end_y, :]

            task_msg = TaskMessage(
                layer_type=layer.type,
                layer_idx=self.current_layer_idx,
                in_channels=layer.in_channels,
                in_h=input_patch.shape[1],
                in_w=input_patch.shape[2],
                out_channels=layer.out_channels,
                out_h=end_row - start_row,
                out_w=W_out,
                kernel_size=layer.kernel_size,
                stride=layer.stride,
                padding=layer.padding,
                groups=layer.groups,
                in_features=0,
                out_features=0,
                input_size=input_patch.size
            )

            task = asyncio.create_task(
                self._send_task_to_worker(worker, task_msg, input_patch)
            )
            tasks.append((worker, start_row, end_row, task))
            logger.debug(f"[Coordinator]: Assigned output rows {start_row}-{end_row} to worker {worker.worker_id} for layer {layer.name}")
        
        await asyncio.gather(*[t[3] for t in tasks])
        output_shape = (layer.out_channels, H_out, W_out)
        self.feature_map = await self._collect_results(tasks, output_shape)

    async def _distribute_fc(self, layer: LayerConfig, quant_params: QuantParams):
        """Split the feature map by output classes"""
        input_vec = self.feature_map.flatten()
        total_classes = layer.out_channels
        available_workers = list(self.worker_manager.workers.values())
        num_workers = len(available_workers) # TODO maybe get idle workers
        classes_per_worker = int(np.ceil(total_classes / num_workers))

        logger.debug(f"[Coordinator]: Distributing FC layer {layer.name} with {total_classes} classes across {num_workers} workers")
        
        tasks = []
        for i, worker in enumerate(available_workers):
            # start_cls = i * classes_per_worker
            # end_cls = min(start_cls + classes_per_worker, total_classes)
            start_cls = worker.worker_id * classes_per_worker
            end_cls = min(start_cls + classes_per_worker, total_classes)
            
            if start_cls >= total_classes:
                continue
            
            task_msg = TaskMessage(
                layer_type=layer.type,
                layer_idx=self.current_layer_idx,
                in_channels=layer.in_channels,
                in_h=1,
                in_w=1,
                out_channels=end_cls - start_cls,
                out_h=1,
                out_w=1,
                kernel_size=0,
                stride=0,
                padding=0,
                groups=0,
                in_features=input_vec.size,
                out_features=end_cls - start_cls,
                input_size=input_vec.size
            )
            task = asyncio.create_task(
                self._send_task_to_worker(worker, task_msg, input_vec)
            )
            tasks.append((worker, start_cls, end_cls, task))
            logger.debug(f"[Coordinator]: Assigned classes {start_cls}-{end_cls} to worker {worker.worker_id} for FC layer {layer.name}")
        await asyncio.gather(*[t[3] for t in tasks])
        
        output_shape = (total_classes,)
        self.feature_map = await self._collect_results(tasks, output_shape)
        
    # TODO need further check
    async def _apply_residual(self, residual_from: str):
        if residual_from not in self.residual_buffers:
            logger.error(f"[Coordinator]: Residual buffer {residual_from} not found for residual connection")
            return
        
        cached, res_s, res_zp = self.residual_buffers[residual_from]
        if cached.shape != self.feature_map.shape:
            logger.error(f"[Coordinator]: Residual buffer shape {cached.shape} does not match current feature map shape {self.feature_map.shape}")
            return
        
        res_f = (cached.astype(np.float32) - res_zp) * res_s

        curr_scale = self.quant_params_list[self.current_layer_idx].s_out
        curr_zero_point = self.quant_params_list[self.current_layer_idx].z_out
        curr_f = (self.feature_map.astype(np.float32) - curr_zero_point) * curr_scale
        
        sum_f = curr_f + res_f
        target_s = self.quant_params_list[self.current_layer_idx].s_residual_out
        target_z = self.quant_params_list[self.current_layer_idx].z_residual_out
        self.feature_map = np.clip(np.round(sum_f / target_s + target_z), 0, 255).astype(np.uint8)

        logger.debug(f"[Coordinator]: Applied residual connection from {residual_from} to current layer {self.current_layer_idx}, feature map updated")

        if self.current_layer_idx == 51:
            logger.debug(f"[Coordinator]: Completed layer {residual_from}, output feature map shape: {self.feature_map.shape}")
            # hex_str = np.array2string(
            #     self.feature_map[1, :, :], 
            #     formatter={'int': lambda x: f'0x{x:02X}'}
            # )

            # logger.debug(f"[Coordinator]: Input for this layer is:\n{padded[1, :1, :]}\n")
            logger.debug(f"[Coordinator]: Sample output hex values:\n{self.feature_map[1, :2, :]}\n")

    async def _send_task_to_worker(self, worker: WorkerInfo, task_msg: TaskMessage, input_patch: np.ndarray):
        worker.state = WorkerState.BUSY
        # Ensure C-contiguous layout before serializing: slicing along axis-1 (e.g. padded[:, a:b, :])
        input_bytes = np.ascontiguousarray(input_patch).tobytes()

        send_start = time.perf_counter()
        await self.worker_manager.send_message(worker, MessageType.TASK, task_msg.pack() + input_bytes)
        send_time = time.perf_counter() - send_start

        # init the worker's stats
        self.current_layer_stats["workers"][worker.worker_id] = {
            "send_time_ms": send_time * 1000,
            "recv_time_ms": 0.0,
            "mcu_compute_ms": 0.0,
        }

        logger.debug(f"[Coordinator]: Sent task for layer {self.current_layer_idx} to worker {worker.worker_id}, waiting for result...")

    async def _collect_results(self, tasks: list[asyncio.Task], output_shape: tuple) -> np.ndarray:
        output = np.zeros(output_shape, dtype=np.uint8)
        num_workers = len(tasks)
        logger.debug(f"[Coordinator]: Collecting results from {num_workers} workers for layer {self.current_layer_idx}")
        
        receive_tasks = []
        for worker, start_idx, end_idx, _ in tasks:
            task = asyncio.create_task(
                self._receive_worker_result(worker, start_idx, end_idx, output)
            )
            receive_tasks.append(task)
        await asyncio.gather(*receive_tasks)
        
        return output
    
    async def _receive_worker_result(self, worker: WorkerInfo, start_idx: int, end_idx: int, output: np.ndarray):
        try:
            #  wait for result message
            header, payload = await self.worker_manager.receive_message(
                worker, 
                timeout=60
            )
            
            if not (header and payload):
                raise RuntimeError(f"Failed to receive result from worker {worker.worker_id}")
            
            if header.type == MessageType.ERROR:
                err_msg = ErrorMessage.unpack(payload)
                logger.error(f"[Coordinator]: Received error from worker {worker.worker_id}: error code: {err_msg.error_code}, message: {err_msg.description}")
                raise RuntimeError(f"error: {err_msg.description}")
            
            if header.type != MessageType.RESULT:
                raise RuntimeError(f"Expected RESULT, got {LayerType(header.type)}")
            
            result_msg = ResultMessage.unpack(payload)
            logger.debug(f"[Coordinator]: result message: {result_msg}")
            
            # read exact output data
            # output_data = await worker.reader.readexactly(result_msg.output_size)
            recv_start = time.perf_counter()
            output_data = await asyncio.wait_for(worker.reader.readexactly(result_msg.output_size), timeout=10)
            recv_time = time.perf_counter() - recv_start

            logger.debug(f"[Coordinator]: Received result header from worker {worker.worker_id} with output size {result_msg.output_size} bytes")
            
            # parse output data and write to the correct position in the output feature map
            if output.ndim == 3:
                # Conv layer: (C, H_slice, W)
                C, _, W = output.shape
                H_slice = end_idx - start_idx
                output_patch = np.frombuffer(output_data, dtype=np.uint8).reshape(
                    (C, H_slice, W)
                )
                logger.debug(f"[Coordinator]: worker:{worker.worker_id}, output_data size: {len(output_data)} bytes, reshaped to {output_patch.shape}")
                output[:, start_idx:end_idx, :] = output_patch
            else:
                # Linear layer: (num_classes,)
                output_patch = np.frombuffer(output_data, dtype=np.uint8)
                output[start_idx:end_idx] = output_patch
            
            # update stats
            # self.stats.total_comm_volume += result_msg.output_size
            # self.stats.total_compute_time += result_msg.compute_time_us / 1e6
            if worker.worker_id in self.current_layer_stats["workers"]:
                ws = self.current_layer_stats["workers"][worker.worker_id]
                ws["mcu_compute_ms"] = result_msg.compute_time_us / 1000
                ws["recv_time_ms"] = recv_time * 1000
            
            # mark worker idle again
            # worker.state = WorkerState.IDLE
            self.worker_manager.mark_worker_idle(worker)
            
            logger.debug(f"[Coordinator]: Received result from worker {worker.worker_id}, "
                        f"slice [{start_idx}, {end_idx}), "
                        f"compute time: {result_msg.compute_time_us / 1000:.2f} ms")
        
        except Exception as e:
            logger.error(f"[Coordinator]: Error receiving result from worker {worker.worker_id}: {e}")
            await self.shutdown_workers() # if any error happens, we shutdown all workers to avoid hanging
            worker.state = WorkerState.DISCONNECTED
            raise
    
    async def shutdown_workers(self):
        logger.info(f"[Coordinator]: Sending shutdown message to all workers")
        shutdown_msg = b'' # no payload needed for shutdown
        for worker in self.worker_manager.workers.values():
            await self.worker_manager.send_message(worker, MessageType.SHUTDOWN, shutdown_msg)
        #  await self.worker_manager.send_message(worker, MessageType.TASK, task_msg.pack() + input_patch.tobytes())


    def _parse_layer_configs(self, json_path: str = './src/model_config.json'):
        with open(json_path, 'r') as f:
            data = json.load(f)
        logger.info(f"[Coordinator]: Loaded model config from {json_path}, total layers: {len(data['layers'])}")
        
        layer_configs = []
        quant_params = []
        for idx, layer_data in enumerate(data["layers"]):
            layer_config_dict = layer_data["layer_config"]
            quant_params_dict = layer_data["quant_params"]

            layer_type = LayerType(layer_config_dict["type"])

            cfg = LayerConfig(
                name=layer_config_dict["name"],
                type=layer_type,
                layer_idx=idx,
                in_channels=layer_config_dict["in_channels"],
                out_channels=layer_config_dict["out_channels"],
                kernel_size=layer_config_dict["kernel_size"],
                stride=layer_config_dict["stride"],
                padding=layer_config_dict["padding"],
                groups=layer_config_dict["groups"],
                residual_add_to=layer_config_dict["residual_add_to"],
                residual_connect_from=layer_config_dict["residual_connect_from"]
            )

            qp = QuantParams(
                s_in=float(quant_params_dict["s_in"]),
                z_in=int(quant_params_dict["z_in"]),
                s_w=np.array(quant_params_dict["s_w"], dtype=np.float32),
                z_w=np.array(quant_params_dict["z_w"], dtype=np.int32),
                s_out=float(quant_params_dict["s_out"]),
                z_out=int(quant_params_dict["z_out"]),
                m=np.array(quant_params_dict["m"], dtype=np.float32),
                s_residual_out=float(quant_params_dict["s_residual_out"]) if quant_params_dict["s_residual_out"] is not None else None,
                z_residual_out=int(quant_params_dict["z_residual_out"]) if quant_params_dict["z_residual_out"] is not None else None
            )

            layer_configs.append(cfg)
            quant_params.append(qp)

        self.layer_config_list = layer_configs
        self.quant_params_list = quant_params
        logger.info(f"[Coordinator]: Parsed {len(self.layer_config_list)} layers and quantization parameters from config")
    
    def _quantize_input(self, input_data: np.ndarray, quant_params: QuantParams) -> np.ndarray:
        s_in = quant_params.s_in
        z_in = quant_params.z_in
        quantized = np.clip(np.round(input_data / s_in + z_in), 0, 255).astype(np.uint8)
        return quantized
    
    def print_stats(self):
        logger.info(f"[Coordinator]: Inference execution stats:")
        for s in self.stats:
            logger.info(
                f"Layer {s['layer_idx']:>3} [{s['layer_type']:>8}] {s['layer_name']}: "
                f"total={s['total_time_ms']:.2f}ms  "
                f"compute={s.get('avg_compute_ms', 0):.2f}ms  "
                # f"comm={s.get('avg_comm_ms', 0):.2f}ms"
            )