import struct
from dataclasses import dataclass
from enum import IntEnum

PROTOCOL_MAGIC = 0xDEADBEEF

class ErrorCode(IntEnum):
    ERR_NONE = 0x00,
    ERR_OUT_OF_MEMORY = 0x01,
    ERR_INVALID_TASK = 0x02,

class MessageType(IntEnum):
    REGISTER = 0x01, # worker -> server
    REGISTER_ACK = 0x02, # server -> worker
    TASK = 0x03, # server -> worker
    RESULT = 0x04, # worker -> server
    ERROR = 0x05, # worker -> server
    HEARTBEAT = 0x06, # worker -> server
    SHUTDOWN = 0x07, # server -> worker

class LayerType(IntEnum):
    CONV = 0x01,
    DEPTHWISE = 0x02,
    POINTWISE = 0x03,
    FC = 0x04,


@dataclass
class MessageHeader:
    FORMAT = '<IBBI6s'
    SIZE = struct.calcsize(FORMAT)


    magic: int = PROTOCOL_MAGIC
    type: MessageType = MessageType.REGISTER    
    worker_id: int = 0
    payload_len: int = 0
    reserved: bytes = bytes(6) # for future use

    def pack(self) -> bytes:
        # little-endian
        return struct.pack(self.FORMAT, self.magic, self.type, self.worker_id, self.payload_len, self.reserved)

    @staticmethod
    def unpack(data: bytes) -> 'MessageHeader':
        if len(data) < 16:
            raise ValueError("Insufficient data for MessageHeader")
        magic, type, worker_id, payload_len, reserved = struct.unpack(MessageHeader.FORMAT, data[:16])
        if magic != PROTOCOL_MAGIC:
            raise ValueError("Invalid magic number")
        return MessageHeader(magic, MessageType(type), worker_id, payload_len, reserved)
    
@dataclass
class RegisterMessage:
    FORMAT = '<I'
    SIZE = struct.calcsize(FORMAT)

    clock_mhz: int
    
    @staticmethod
    def unpack(data: bytes) -> 'RegisterMessage':
        if len(data) < RegisterMessage.SIZE:
            raise ValueError("Insufficient data for RegisterMessage")
        clock_mhz, = struct.unpack(RegisterMessage.FORMAT, data[:RegisterMessage.SIZE])
        return RegisterMessage(clock_mhz)
    
@dataclass
class RegisterAckMessage:
    FORMAT = '<BB'
    SIZE = struct.calcsize(FORMAT)

    status: int
    assigned_id: int

    def pack(self) -> bytes:
        return struct.pack(RegisterAckMessage.FORMAT, self.status, self.assigned_id)

# TODO optimize the payload structure, e.g. conv params and linear params don't need to be transmitted in the task message
@dataclass
class TaskMessage:
    FORMAT = '<BIIIIIIIBBBHIII'
    SIZE = struct.calcsize(FORMAT)
    
    layer_type: LayerType
    layer_idx: int
    # input/output channels and dimensions
    in_channels: int
    in_h: int
    in_w: int

    out_channels: int
    out_h: int
    out_w: int

    # convolution parameters
    kernel_size: int
    stride: int
    padding: int
    groups: int

    # linear parameters
    in_features: int
    out_features: int

    # data
    # TODO in bytes, the real data will be sent after the header and TaskMessage
    # But really???
    input_size: int 

    def pack(self) -> bytes:
        data = struct.pack('<BI', self.layer_type, self.layer_idx)
        data += struct.pack('<IIIIII', self.in_channels, self.in_h, self.in_w, self.out_channels, self.out_h, self.out_w)
        data += struct.pack('<BBBH', self.kernel_size, self.stride, self.padding, self.groups)
        data += struct.pack('<III', self.in_features, self.out_features, self.input_size)
        return data


@dataclass
class ResultMessage:
    FORMAT = '<II'
    SIZE = struct.calcsize(FORMAT)
    
    compute_time_us: int
    output_size: int # in bytes

    @staticmethod
    def unpack(data: bytes) -> 'ResultMessage':
        if len(data) < ResultMessage.SIZE:
            raise ValueError("Insufficient data for ResultMessage")
        compute_time_us, output_size = struct.unpack(ResultMessage.FORMAT, data[:ResultMessage.SIZE])
        return ResultMessage(compute_time_us, output_size)



@dataclass
class ErrorMessage:
    FORMAT = '<B63s'
    SIZE = struct.calcsize(FORMAT)

    error_code: int
    description: str
    
    @staticmethod
    def unpack(data: bytes) -> 'ErrorMessage':
        if len(data) < ErrorMessage.SIZE:
            raise ValueError("Insufficient data for ErrorMessage")
        error_code, description = struct.unpack(ErrorMessage.FORMAT, data[:ErrorMessage.SIZE])
        return ErrorMessage(error_code, description.decode('utf-8').rstrip('\x00'))