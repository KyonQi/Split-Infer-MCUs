#include "worker.h"
#include "protocol.h"
#include "weights.h"
#include "layer_config.h"
#include "quant_params.h"
#include "conv/conv2d.h"
#include "linear/linear.h"
#include "rans.h"

uint8_t Worker::input_buffer_[350 * 1024];  // RAM1: 350KB
DMAMEM uint8_t Worker::output_buffer_[350 * 1024];  // RAM2: 350KB

Worker::Worker(uint8_t worker_id, IPAddress svr_ip, uint16_t svr_port)
    : worker_id_(worker_id), svr_ip_(svr_ip), svr_port_(svr_port), is_connected_(false) {
    state_ = WorkerState::DISCONNECTED;
}

Worker::~Worker() {
    if (client_.connected()) {
        client_.stop();
    }
}

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
#ifdef DEBUG
    Serial.printf("Worker %d idle, waiting for tasks...\n", worker_id_);
#endif
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
            state_ = WorkerState::DISCONNECTED;
            return;
        }
    }
}

void Worker::HandleReceivingTask() {
#ifdef DEBUG
    Serial.printf("Worker %d receiving task...\n", worker_id_);
#endif
    Read((uint8_t *)&current_task_, sizeof(current_task_)); // TODO error 
    
    // add the halo reuse logic
    if (current_task_.use_halo_cache) {
        // halo reuse
        if (!cache_valid_ || cache_block_end_idx_ != current_task_.prev_block_end_idx) {
            SendError(ErrorCode::ERR_INVALID_TASK, "Invalid halo cache state for reuse");
            state_ = WorkerState::IDLE;
            return;
        }
        
        uint16_t top_rows = current_task_.halo_top_rows;
        uint16_t bottom_rows = current_task_.halo_bottom_rows;
        uint16_t cache_use_start = current_task_.cache_use_start;
        uint16_t cache_use_end = current_task_.cache_use_end;
        uint16_t cache_use_rows = cache_use_end - cache_use_start;
        
        uint16_t C = cache_channels_;
        uint16_t W = cache_width_;
        uint16_t H_total = top_rows + cache_use_rows + bottom_rows;

        // layout: [C, (halo_top, cache_use, halo_bottom), W]
        size_t top_bytes = (size_t)C * top_rows * W;
        size_t cache_bytes = (size_t)C * cache_use_rows * W;
        size_t bottom_bytes = (size_t)C * bottom_rows * W;

        // halo_top and halo_bottom will be placed at the end of buffer first
        size_t tmp_top_off = sizeof(input_buffer_) - top_bytes - bottom_bytes;
        size_t tmp_bottom_off = sizeof(input_buffer_) - bottom_bytes;
        if (top_bytes) Read(input_buffer_ + tmp_top_off, top_bytes);
        if (bottom_bytes) Read(input_buffer_ + tmp_bottom_off, bottom_bytes);
#ifdef DEBUG
        Serial.printf("Worker %d received task with halo reuse, top_rows: %d, bottom_rows: %d, cache_use_rows: %d\n", 
            worker_id_, top_rows, bottom_rows, cache_use_rows);
#endif
        // copy by channels
        for (uint16_t c = 0; c < C; ++c) {
            uint8_t *dst = input_buffer_ + (size_t)c * H_total * W;
            const uint8_t *cache_row = output_buffer_ + ((size_t)c * cache_rows_ + cache_use_start) * W;

            if (top_rows) {
                memcpy(dst, input_buffer_ + tmp_top_off + (size_t)c * top_rows * W, (size_t)top_rows * W);
            }
            memcpy(dst + (size_t)top_rows * W, cache_row, (size_t)cache_use_rows * W);
            if (bottom_rows) {
                memcpy(dst + (size_t)(top_rows + cache_use_rows) * W, 
                        input_buffer_ + tmp_bottom_off + (size_t)c * bottom_rows * W, 
                        (size_t)bottom_rows * W);
            }
        }
            

    } else {
        // normal path
        uint32_t total_data_size = current_task_.input_size;
        if (total_data_size > sizeof(input_buffer_)) {
            Serial.println("Input data size exceeds buffer size");
            SendError(ErrorCode::ERR_OUT_OF_MEMORY, "Input data size exceeds buffer size");
            state_ = WorkerState::IDLE;
            return;
        }
        Read(input_buffer_, total_data_size);

        // Auto-detect rANS compressed input and decompress
        if (total_data_size >= sizeof(rans::Header)) {
            const rans::Header *hdr = reinterpret_cast<const rans::Header *>(input_buffer_);
            if (hdr->magic == rans::MAGIC) {
                // Decompress: input_buffer_ → output_buffer_ → input_buffer_
                size_t decompressed = rans::decompress(input_buffer_, total_data_size,
                                                    output_buffer_, sizeof(output_buffer_));
                if (decompressed == 0) {
                    SendError(ErrorCode::ERR_OUT_OF_MEMORY, "Input decompression failed");
                    state_ = WorkerState::IDLE;
                    return;
                }
                memcpy(input_buffer_, output_buffer_, decompressed);
#ifdef DEBUG
                Serial.printf("Worker %d decompressed input: %u -> %u bytes\n",
                            worker_id_, total_data_size, (uint32_t)decompressed);
#endif
            }
        }
    }
    state_ = WorkerState::COMPUTING;
}

// TODO need further developments
void Worker::HandleComputing() {
#ifdef DEBUG
    Serial.printf("Worker %d processing task %d...\n", worker_id_, static_cast<uint8_t>(current_task_.layer_type));
#endif

    // Block mode: multiple consecutive layers
    if (current_task_.end_layer_idx >= current_task_.layer_idx) {
        HandleBlockComputing();
        return;
    }

    // Single layer mode (original path)
    int layer_idx = current_task_.layer_idx;
    bool success = false;
    uint8_t *input = input_buffer_;
    const int8_t *weights = model_weights[layer_idx].weights;
    const int32_t *bias = model_weights[layer_idx].bias;
    uint8_t *output = output_buffer_;

#ifdef DEBUG
    // Memory tracker: single layer → layer_in_block = 0, src=RAM1, dst=RAM2
    mem_tracker_.Reset(sizeof(input_buffer_), sizeof(output_buffer_));
    uint32_t input_data_bytes  = current_task_.in_channels * current_task_.in_h * current_task_.in_w;
    uint32_t output_data_bytes = current_task_.out_channels * current_task_.out_h * current_task_.out_w;
    mem_tracker_.RecordBeforeLayer(0, layer_idx, input_data_bytes, output_data_bytes,
                                   model_weights[layer_idx].weights_size);
    uint32_t *paint_base = mem::paintStackRegion();
#endif

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
            success = true;
            break;
        case LayerType::FC:
            linear::native_linear(input, weights, bias, output,
                                    &model_layer_config[layer_idx], &model_quant_params[layer_idx]);
            success = true;
            break;
        default:
            break;
    }
    // TODO check if we need ReLU6 here; QuantWorker doesn't have it
    uint32_t task_elapsed_time = micros() - task_start_time;

#ifdef DEBUG
    uint32_t stack_peak = mem::checkStackWaterMark(paint_base);
    mem_tracker_.RecordAfterLayer(0, stack_peak);
    // mem_tracker_.PrintReport();
    mem_tracker_.PrintLayerDetail();
#endif

    if (!success) {
        Serial.println("Invalid layer type in task");
        SendError(ErrorCode::ERR_INVALID_TASK, "Invalid layer type in task");
        state_ = WorkerState::IDLE;
        return;
    }

    
    // rANS compression: output_buffer_ → input_buffer_ (cannot compress in-place)
//     uint32_t raw_output_size = current_task_.out_channels * current_task_.out_h * current_task_.out_w;
//     uint32_t compress_start = micros();
//     size_t compress_size = rans::compress(output_buffer_, raw_output_size, input_buffer_, sizeof(input_buffer_));
//     uint32_t compress_time = micros() - compress_start;
//     if (0 == compress_size) {
//         Serial.println("Compression failed");
//         SendError(ErrorCode::ERR_OUT_OF_MEMORY, "Compression failed");
//         state_ = WorkerState::IDLE;
//         return;
//     }
// #ifdef DEBUG
//     Serial.printf("Worker %d finished computing, raw output size: %d bytes, compressed size: %d bytes, compute time: %d us, compression time: %d us\n", 
//         worker_id_, raw_output_size, compress_size, task_elapsed_time, compress_time);
// #endif


    current_result_.compute_time_us = task_elapsed_time;
    current_result_.compress_time_us = 0;
    // current_result_.output_size = static_cast<uint32_t>(compress_size);
    current_result_.output_size = current_task_.out_channels * current_task_.out_h * current_task_.out_w; // TODO need further check if we can directly send raw output without compression
    state_ = WorkerState::SENDING_RESULT;
}

void Worker::HandleBlockComputing() {
    uint32_t start_idx = current_task_.layer_idx;
    uint32_t end_idx = current_task_.end_layer_idx;

#ifdef DEBUG
    // Serial.printf("Worker %d block computing layers %u -> %u\n", worker_id_, start_idx, end_idx);
    mem_tracker_.Reset(sizeof(input_buffer_), sizeof(output_buffer_));
#endif

    uint8_t *src = input_buffer_;
    uint8_t *dst = output_buffer_;

    uint32_t cur_h = current_task_.in_h;
    uint32_t cur_w = current_task_.in_w;
    uint32_t total_compute_us = 0;

    for (uint32_t idx = start_idx; idx <= end_idx; idx++) {
        const LayerConfig *cfg = &model_layer_config[idx];
        const int8_t *weights = model_weights[idx].weights;
        const int32_t *bias = model_weights[idx].bias;

        // Infer layer type from config: depthwise has in_channels == out_channels and kernel > 1
        bool is_depthwise = (cfg->input_channels == cfg->output_channels && cfg->kernel_size > 1);
        bool is_pointwise = (cfg->kernel_size == 1);

#ifdef DEBUG
        uint32_t out_h, out_w;
        if (is_depthwise) {
            uint8_t pt = current_task_.block_pad_top;
            uint8_t pb = current_task_.block_pad_bottom;
            uint8_t pl = cfg->padding;
            uint8_t pr = cfg->padding;
            out_h = (cur_h + pt + pb - cfg->kernel_size) / cfg->stride + 1;
            out_w = (cur_w + pl + pr - cfg->kernel_size) / cfg->stride + 1;
        } else {
            // Single-layer task: coordinator computed per-worker H padding
            // Multi-layer block: non-DW layers (expand/project 1x1) use cfg->padding (= 0)
            uint8_t pt = (start_idx == end_idx) ? current_task_.block_pad_top : cfg->padding;
            uint8_t pb = (start_idx == end_idx) ? current_task_.block_pad_bottom : cfg->padding;
            uint8_t pl = cfg->padding;
            uint8_t pr = cfg->padding;

            out_h = (cur_h + pt + pb - cfg->kernel_size) / cfg->stride + 1;
            out_w = (cur_w + pl + pr - cfg->kernel_size) / cfg->stride + 1;
        }

        uint32_t input_data_bytes  = cfg->input_channels * cur_h * cur_w;
        uint32_t output_data_bytes = cfg->output_channels * out_h * out_w;
        mem_tracker_.RecordBeforeLayer(idx - start_idx, idx, input_data_bytes, output_data_bytes, model_weights[idx].weights_size);
        uint32_t *paint_base = mem::paintStackRegion();
#endif

        uint32_t t0 = micros();

        if (is_depthwise) {
            // Worker-side DW padding: use block_pad_top/bottom for height,
            // cfg->padding for width.  The padded DW kernel handles padding
            // via bounds-checking, which is exact with z_in_dw.
            uint8_t pt = current_task_.block_pad_top;
            uint8_t pb = current_task_.block_pad_bottom;
            uint8_t pl = cfg->padding;
            uint8_t pr = cfg->padding;
            conv2d::depthwise_conv2d_padded(src, weights, bias, dst,
                                     cfg, &model_quant_params[idx],
                                     cur_h, cur_w, pt, pb, pl, pr);
            cur_h = (cur_h + pt + pb - cfg->kernel_size) / cfg->stride + 1;
            cur_w = (cur_w + pl + pr - cfg->kernel_size) / cfg->stride + 1;
        } else if (is_pointwise) {
            uint32_t out_size = cfg->output_channels * cur_h * cur_w;
            uint8_t *workspace = dst + out_size; // use dst tail as workspace, assuming it's large enough for pointwise conv
            uint32_t workspace_size = sizeof(output_buffer_) - out_size;
            
            if (workspace_size < (cur_h * cur_w + 1) * cfg->input_channels * 2) { // x2: pointwise workspace uses q15_t (2 bytes)
#ifdef DEBUG
                Serial.println("Not enough workspace for pointwise convolution, back to native conv2d without workspace");
#endif
                // Single-layer task: coordinator computed per-worker H padding
                // Multi-layer block: non-DW layers (expand/project 1x1) use cfg->padding (= 0)
                uint8_t pt = (start_idx == end_idx) ? current_task_.block_pad_top : cfg->padding;
                uint8_t pb = (start_idx == end_idx) ? current_task_.block_pad_bottom : cfg->padding;
                uint8_t pl = cfg->padding;
                uint8_t pr = cfg->padding;
                conv2d::native_conv2d_padded(src, weights, bias, dst,
                                        cfg, &model_quant_params[idx],
                                        cur_h, cur_w, pt, pb, pl, pr);

                cur_h = (cur_h + pt + pb - cfg->kernel_size) / cfg->stride + 1;
                cur_w = (cur_w + pl + pr - cfg->kernel_size) / cfg->stride + 1;
            } else {
                conv2d::pointwise_conv2d(src, weights, bias, dst,
                                        cfg, &model_quant_params[idx], cur_h, cur_w,
                                        workspace, workspace_size);                
            }
            // conv2d::pointwise_conv2d(src, weights, bias, dst,
            //                         cfg, &model_quant_params[idx], cur_h, cur_w,
            //                         workspace, workspace_size);
             // pointwise conv does not change H/W
        } else {
            // Single-layer task: coordinator computed per-worker H padding
            // Multi-layer block: non-DW layers (expand/project 1x1) use cfg->padding (= 0)
            uint8_t pt = (start_idx == end_idx) ? current_task_.block_pad_top : cfg->padding;
            uint8_t pb = (start_idx == end_idx) ? current_task_.block_pad_bottom : cfg->padding;
            uint8_t pl = cfg->padding;
            uint8_t pr = cfg->padding;
            conv2d::native_conv2d_padded(src, weights, bias, dst,
                                     cfg, &model_quant_params[idx],
                                     cur_h, cur_w, pt, pb, pl, pr);

            cur_h = (cur_h + pt + pb - cfg->kernel_size) / cfg->stride + 1;
            cur_w = (cur_w + pl + pr - cfg->kernel_size) / cfg->stride + 1;
        }
        total_compute_us += micros() - t0;

#ifdef DEBUG
        // Serial.printf("  Layer %u (%s): -> %ux%u\n",
        //               idx, cfg->name, cur_h, cur_w);
        uint32_t stack_peak = mem::checkStackWaterMark(paint_base);
        mem_tracker_.RecordAfterLayer(idx - start_idx, stack_peak);
#endif

        // Ping-pong: swap src and dst
        uint8_t *tmp = src;
        src = dst;
        dst = tmp;
    }

    // After the loop, src points to the last output (swapped after last layer write to dst)
    // For odd number of layers: src = output_buffer_  (good, HandleSendingResult reads output_buffer_)
    // For even number of layers: src = input_buffer_  (need to copy to output_buffer_)
    uint32_t num_layers = end_idx - start_idx + 1;
    uint32_t output_size = current_task_.out_channels * current_task_.out_h * current_task_.out_w;

    if (num_layers % 2 == 0) {
        // Final output is in input_buffer_, copy to output_buffer_
        memcpy(output_buffer_, input_buffer_, output_size);
    }

    // halo reuse flag
    cache_valid_ = true;
    cache_block_end_idx_ = end_idx;
    cache_channels_ = current_task_.out_channels;
    cache_rows_ = current_task_.out_h;
    cache_width_ = current_task_.out_w;

#ifdef DEBUG
    // Serial.printf("Worker %d block done: %u layers, output %u bytes, compute %u us\n",
    //               worker_id_, num_layers, output_size, total_compute_us);

    // mem_tracker_.PrintReport();
    mem_tracker_.PrintLayerDetail();
#endif

    current_result_.compute_time_us = total_compute_us;
    current_result_.compress_time_us = 0;
    current_result_.output_size = output_size;
    state_ = WorkerState::SENDING_RESULT;
}

void Worker::HandleSendingResult() {
#ifdef DEBUG
    Serial.printf("Worker %d sending result...\n", worker_id_);
#endif
    MessageHeader header;
    init_header(header, MessageType::RESULT, worker_id_, sizeof(ResultMessage));

    Send((const uint8_t *)&header, sizeof(header));
    Send((const uint8_t *)&current_result_, sizeof(current_result_));

    // send big data in chunks
    const size_t CHUNK_SIZE = 1024;  // 1KB per chunk, can be tuned based on performance testing
    size_t total = current_result_.output_size;
    size_t offset = 0;
    
    while (offset < total) {
        size_t chunk = min(CHUNK_SIZE, total - offset);
        // Send((const uint8_t *)&input_buffer_[offset], chunk);
        Send((const uint8_t *)&output_buffer_[offset], chunk); // if not compress
        offset += chunk;
    }

    // Send(output_buffer_, current_result_.output_size);
    client_.flush();
#ifdef DEBUG
    Serial.printf("Worker %d finish sending...\n", worker_id_);
#endif
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
#ifdef DEBUG
    Serial.printf("Worker %d sent error message: %s\n", worker_id_, err_msg.description);
#endif
}


void Worker::Send(const uint8_t *buffer, size_t size) {
    size_t bytes_sent = 0;
    while (bytes_sent < size) {
        int n = client_.write(buffer + bytes_sent, size - bytes_sent);
        if (n > 0) {
            bytes_sent += n;
        } else if (n == 0) {
            // buffer is full, wait for it to drain before sending more
            if (!client_.connected()) {
                Serial.println("Connection lost while sending");
                break;
            }
            delay(1);
        } else {
            // n < 0 means an error occurred
            Serial.println("Error sending data to server");
            break;
        }
    }
}

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

