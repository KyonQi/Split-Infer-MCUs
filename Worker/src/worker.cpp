#include "worker.h"
#include "protocol.h"
#include "weights.h"
#include "layer_config.h"
#include "quant_params.h"
#include "conv/conv2d.h"
#include "linear/linear.h"

uint8_t Worker::input_buffer_[350 * 1024];  // RAM1: 350KB
DMAMEM uint8_t Worker::output_buffer_[350 * 1024];  // RAM2: 350KB

Worker::Worker(uint8_t worker_id, IPAddress svr_ip, uint16_t svr_port)
    : worker_id_(worker_id), svr_ip_(svr_ip), svr_port_(svr_port), is_connected_(false) {
    state_ = WorkerState::DISCONNECTED;
}

Worker::~Worker() {}

void Worker::Begin() {
    // allocate static IP
    uint8_t mac[6] = { 0xDE, 0xAD, 0xBE, 0xEF, 0xFE, worker_id_ };
    IPAddress local_ip(192, 168, 1, 110 + worker_id_);  // cutomize it to avoid IP conflicts
    IPAddress dns(192, 168, 1, 1);
    IPAddress gateway(192, 168, 1, 1);
    IPAddress subnet(255, 255, 255, 0);

    Ethernet.begin(mac, local_ip, dns, gateway, subnet);
    
    // wait some time to init
    delay(1000);
    
    Serial.printf("Worker %d started with IP: %d.%d.%d.%d\n", 
        worker_id_, local_ip[0], local_ip[1], local_ip[2], local_ip[3]);

    // ConnectToServer();
}

void Worker::Loop() {
    switch (state_) {
    case WorkerState::DISCONNECTED:
        HandleDisconnected();
        break;
    case WorkerState::CONNECTING:
        HandleConnecting();
        break;
    case WorkerState::REGISTERING:
        HandleRegistering();
        break;
    case WorkerState::IDLE:
        HandleIdle();
        break;
    case WorkerState::RECEIVING_TASK:
        HandleReceivingTask();
        break;
    case WorkerState::COMPUTING:
        HandleComputing();
        break;
    case WorkerState::SENDING_RESULT:
        HandleSendingResult();
        break;
    default:
        break;
    }
}

void Worker::HandleDisconnected() {
    state_ = WorkerState::CONNECTING;
}

void Worker::HandleConnecting() {
    Serial.printf("Worker %d connecting to server %d.%d.%d.%d:%d...\n", 
        worker_id_, svr_ip_[0], svr_ip_[1], svr_ip_[2], svr_ip_[3], svr_port_);

    if (client_.connect(svr_ip_, svr_port_)) {
        Serial.printf("Worker %d connected to server %d.%d.%d.%d:%d\n", 
            worker_id_, svr_ip_[0], svr_ip_[1], svr_ip_[2], svr_ip_[3], svr_port_);
        is_connected_ = true;
        state_ = WorkerState::REGISTERING;
        return; 
    }
    Serial.printf("Worker %d failed to connect, retrying in 5s...\n", worker_id_);
    delay(5000);
}

void Worker::HandleRegistering() {
    SendRegistration();
    
    // wait for registration ack, timeout = 5s
    MessageHeader header;
    uint32_t start_time = millis();
    while (millis() - start_time < 5000) { // wait for 5 seconds
        if (client_.available() >= sizeof(MessageHeader)) {
            Read((uint8_t *)&header, sizeof(header));
            if (header.magic != PROTOCOL_MAGIC || header.type != MessageType::REGISTER_ACK) {
                Serial.printf("Worker %d receive: 0x%08x, type: %d\n", worker_id_, header.magic, header.type);
                Serial.println("Invalid registration ack received, ignoring...");
                continue;
            }
            RegisterAckMessage ack_msg;
            if (header.payload_len != sizeof(RegisterAckMessage)) {
                Serial.println("Invalid registration ack payload length, ignoring...");
                continue;
            }
            Read((uint8_t *)&ack_msg, sizeof(ack_msg)); // TODO error handling
            
            if (ack_msg.status != 0) {
                Serial.printf("Registration failed with error code %d\n", ack_msg.status);
                
                client_.stop(); // TODO how to gracefully abstract the codes here?
                is_connected_ = false;
                state_ = WorkerState::DISCONNECTED;
                return;
            }
            Serial.printf("Worker %d registered successfully with assigned ID %d\n", worker_id_, ack_msg.assigned_id);
            state_ = WorkerState::IDLE;
            return;
        }
    }
    Serial.printf("Worker %d registration timed out, disconnecting...\n", worker_id_);
    client_.stop(); // TODO how to gracefully abstract the codes here?
    is_connected_ = false;
    state_ = WorkerState::DISCONNECTED;
}

void Worker::SendRegistration() {
    MessageHeader header;
    RegisterMessage reg_msg;
    memset(&reg_msg, 0, sizeof(reg_msg));
    // maybe needs refactor to avoid the memcpy
    init_header(header, MessageType::REGISTER, worker_id_, sizeof(RegisterMessage));
    reg_msg.clock_mhz = F_CPU / 1000000;
    Send((const uint8_t *)&header, sizeof(header));
    Send((const uint8_t *)&reg_msg, sizeof(reg_msg)); // TODO error handling needs!
    Serial.printf("Worker %d sent registration message\n", worker_id_);
}

void Worker::HandleIdle() {
    // check if there is a new task
    Serial.printf("Worker %d idle, waiting for tasks...\n", worker_id_);
    if (client_.available() >= sizeof(MessageHeader)) {
        MessageHeader header;
        Read((uint8_t *)&header, sizeof(header)); // TODO notice we need nonblocking way here; also error handling maybe
        if (!validate_header(header)) {
            Serial.println("Invalid message header received, ignoring...");
            return;
        }
        if (header.type == MessageType::TASK) {
            state_ = WorkerState::RECEIVING_TASK;
            return;
        }
        if (header.type == MessageType::SHUTDOWN) {
            client_.stop();
            is_connected_ = false;
            // Serial.printf("Worker %d received shutdown message, disconnecting...\n", worker_id_);
            state_ = WorkerState::DISCONNECTED;
            return;
        }
        // TODO maybe adds shutdown message here
    }
}

void Worker::HandleReceivingTask() {
    Serial.printf("Worker %d receiving task...\n", worker_id_);
    Read((uint8_t *)&current_task_, sizeof(current_task_)); // TODO error handling
    uint32_t total_data_size = current_task_.input_size;
    if (total_data_size > sizeof(input_buffer_)) {
        Serial.println("Input data size exceeds buffer size");
        SendError(ErrorCode::ERR_OUT_OF_MEMORY, "Input data size exceeds buffer size");
        state_ = WorkerState::IDLE;
        return;
    }
    Read(input_buffer_, total_data_size);
    state_ = WorkerState::COMPUTING;
}

// TODO need further developments
void Worker::HandleComputing() {
    Serial.printf("Worker %d processing task %d...\n", worker_id_, static_cast<uint8_t>(current_task_.layer_type));
    // uint32_t start_time = micros();
    int layer_idx = current_task_.layer_idx; // TODO need to get from task payload
    bool success = false;
    uint8_t *input = input_buffer_;
    const int8_t *weights = model_weights[layer_idx].weights;
    const int32_t *bias = model_weights[layer_idx].bias;
    uint8_t *output = output_buffer_;
    uint32_t task_start_time = micros();
    switch (current_task_.layer_type) {
        case LayerType::CONV:
            conv2d::native_conv2d(input, weights, bias, output, 
                                    &model_layer_config[layer_idx], &model_quant_params[layer_idx],
                                    current_task_.in_h, current_task_.in_w);
            success = true;
            break;
        case LayerType::DEPTHWISE:
            conv2d::depthwise_conv2d(input, weights, bias, output, 
                                    &model_layer_config[layer_idx], &model_quant_params[layer_idx],
                                    current_task_.in_h, current_task_.in_w);
        case LayerType::FC:
            linear::native_linear(input, weights, bias, output,
                                    &model_layer_config[layer_idx], &model_quant_params[layer_idx]);
            success = true;
            break;
        default:
            break;
    }
    uint32_t task_elapsed_time = micros() - task_start_time;
    if (!success) {
        Serial.println("Invalid layer type in task");
        SendError(ErrorCode::ERR_INVALID_TASK, "Invalid layer type in task");
        state_ = WorkerState::IDLE;
        return;
    }
    // uint32_t compute_time = micros() - start_time;
    current_result_.compute_time_us = task_elapsed_time;
    current_result_.output_size = current_task_.out_channels * current_task_.out_h * current_task_.out_w; // TODO need to check the actual output size
    state_ = WorkerState::SENDING_RESULT;
}

void Worker::HandleSendingResult() {
    Serial.printf("Worker %d sending result...\n", worker_id_);
    MessageHeader header;
    init_header(header, MessageType::RESULT, worker_id_, sizeof(ResultPayload));

    Send((const uint8_t *)&header, sizeof(header));
    Send((const uint8_t *)&current_result_, sizeof(current_result_));

    // debug printf the output data in hex
    Serial.printf("Output data (hex): ");
    for (size_t i = 0; i < current_result_.output_size && i < 64; i++) { // only print the first 64 bytes for debug
        Serial.printf("%02X ", output_buffer_[i]);
    }
    Serial.println();

    // 3. 分块发送大数据
    const size_t CHUNK_SIZE = 1024;  // 每次发 1KB，适配嵌入式 TCP 缓冲区
    size_t total = current_result_.output_size;
    size_t offset = 0;
    
    while (offset < total) {
        size_t chunk = min(CHUNK_SIZE, total - offset);
        Send((const uint8_t *)&output_buffer_[offset], chunk);
        offset += chunk;
    }

    // Send(output_buffer_, current_result_.output_size);
    client_.flush();
    Serial.printf("Worker %d finish sending...\n", worker_id_);
    
    state_ = WorkerState::IDLE;
}

void Worker::SendError(ErrorCode code, const char *description) {
    MessageHeader header;
    ErrorMessage err_msg;
    memset(&err_msg, 0, sizeof(err_msg));
    strncpy(err_msg.description, description, sizeof(err_msg.description) - 1);
    err_msg.description[sizeof(err_msg.description) - 1] = '\0'; // ensure null-termination
    init_header(header, MessageType::ERROR, worker_id_, sizeof(ErrorMessage));
    err_msg.error_code = static_cast<uint8_t>(code);
    Send((const uint8_t *)&header, sizeof(header));
    Send((const uint8_t *)&err_msg, sizeof(err_msg));
    Serial.printf("Worker %d sent error message: %s\n", worker_id_, err_msg.description);
}


void Worker::Send(const uint8_t *buffer, size_t size) {
    size_t bytes_sent = 0;
    while (bytes_sent < size) {
        int n = client_.write(buffer + bytes_sent, size - bytes_sent);
        if (n > 0) {
            bytes_sent += n;
        } else if (n == 0) {
            // 缓冲区暂时满了，yield 让出 CPU，等待缓冲区腾出空间
            // 不要 break！
            if (!client_.connected()) {
                Serial.println("Connection lost while sending");
                break;
            }
            delay(1);  // 或者 yield(); 取决于你的 RTOS/平台
        } else {
            // n < 0 才是真正的错误
            Serial.println("Error sending data to server");
            break;
        }
    }
    // 不要在这里 flush()，或者只在最后一帧结束后 flush 一次
}

// void Worker::Send(const uint8_t *buffer, size_t size) {
//     size_t bytes_sent = 0;
//     while (bytes_sent < size) {
//         int n = client_.write(buffer + bytes_sent, size - bytes_sent);
//         if (n > 0) {
//             bytes_sent += n;
//         } else {
//             Serial.println("Error sending data to server");
//             break;
//         }
//     }
//     // if (client_.connected()) {
//     //     client_.write(buffer, size);
//     // }
//     client_.flush();
// }

// TODO blocking read, how to unblocking?
void Worker::Read(uint8_t *buffer, size_t size) {
    size_t bytes_read = 0;
    while (bytes_read < size) {
        if (client_.available()) {
            int ret = client_.read(buffer + bytes_read, size - bytes_read);
            if (ret > 0) {
                bytes_read += ret;
            } else {
                Serial.println("Error reading from server");
                break;
            }
        }
    }
}



