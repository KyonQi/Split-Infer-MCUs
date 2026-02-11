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