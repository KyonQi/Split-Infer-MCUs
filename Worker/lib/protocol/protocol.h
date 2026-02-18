#ifndef PROTOCOL_H
#define PROTOCOL_H

#include <Arduino.h>
#include <stdint.h>

#define PROTOCOL_MAGIC 0xDEADBEEF

enum class ErrorCode : uint8_t {
    ERR_NONE = 0x00,
    ERR_OUT_OF_MEMORY = 0x01,
    ERR_INVALID_TASK = 0x02,
};

enum class MessageType : uint8_t {
    REGISTER = 0x01, // worker -> server
    REGISTER_ACK = 0x02, // server -> worker
    TASK = 0x03, // server -> worker
    RESULT = 0x04, // worker -> server
    ERROR = 0x05, // worker -> server
    HEARTBEAT = 0x06, // worker -> server option TODO
    SHUTDOWN = 0x07, // server -> worker
};

// TODO need further check
enum class LayerType : uint8_t {
    CONV = 0x01,
    DEPTHWISE = 0x02,
    POINTWISE = 0x03,
    FC = 0x04,
};

struct MessageHeader {
    uint32_t magic; // fixed value 0xDEADBEEF
    MessageType type;
    uint8_t worker_id;
    uint32_t payload_len;
    uint8_t reserved[6]; // for future use
} __attribute__((packed)); // TODO need further check the attribute; 16 bytes for header

// TODO Need to rename it to RegisterPayload
struct RegisterMessage {
    uint32_t clock_mhz;
} __attribute__((packed)); // TODO need further check the attribute; 24 bytes for payload

// TODO Need to rename it to RegisterAckPayload
struct RegisterAckMessage {
    uint8_t status; // 0 for success, non-zero for error code
    uint8_t assigned_id; // it should be the same as worker_id in header, but server can reassign if needed
} __attribute__((packed)); // TODO need further check the attribute; 1 byte for payload

struct TaskMessage {
    LayerType layer_type;
    uint32_t layer_idx;

    // input/output channels
    uint32_t in_channels, in_h, in_w;
    uint32_t out_channels, out_h, out_w;

    // convolution parameters
    uint8_t kernel_size, stride, padding;
    uint16_t groups;

    // linear parameters
    uint32_t in_features, out_features;

    // data size
    uint32_t input_size; // in bytes    
} __attribute__((packed)); // TODO need further check the attribute; 29 bytes for payload

struct ResultMessage {
    uint32_t compute_time_us;
    uint32_t output_size; // in bytes
    // maybe performance records here
} __attribute__((packed)); // TODO need further check the attribute; 8 bytes for payload

struct ErrorMessage {
    uint8_t error_code;
    char description[63];
} __attribute__((packed)); // TODO need further check the attribute; 64 bytes for payload

inline uint32_t init_header(MessageHeader &header, MessageType type, uint8_t worker_id, uint32_t payload_len) {
    header.magic = PROTOCOL_MAGIC;
    header.type = type;
    header.worker_id = worker_id;
    header.payload_len = payload_len;
    memset(header.reserved, 0, sizeof(header.reserved));
    return sizeof(MessageHeader);
}

inline bool validate_header(const MessageHeader &header) {
    if (header.magic != PROTOCOL_MAGIC) {
        return false;
    }
    return true;
}

#endif // PROTOCOL_H