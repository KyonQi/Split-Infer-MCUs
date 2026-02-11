#ifndef WORKER_H
#define WORKER_H

#include <Arduino.h>
#include <NativeEthernet.h>

#include "protocol.h"

class Worker final {
public:
    Worker(uint8_t worker_id, IPAddress svr_ip, uint16_t svr_port);

    ~Worker();

    void Begin(); // simiar to setup()

    void Loop();

private:
    enum class WorkerState : uint8_t {
        DISCONNECTED,
        CONNECTING,
        REGISTERING,
        IDLE,
        RECEIVING_TASK,
        COMPUTING,
        SENDING_RESULT,
    };

private:
    void HandleDisconnected();
    void HandleConnecting();
    void HandleRegistering();
    void HandleIdle();
    void HandleReceivingTask();
    void HandleComputing();
    void HandleSendingResult();

private:
    void ConnectToServer();
    void SendRegistration(); // maybe no need
    void SendError(ErrorCode code, const char *description);
    void Send(const uint8_t *buffer, size_t size);
    void Read(uint8_t *buffer, size_t size);

private:
    WorkerState state_;
    uint8_t worker_id_;
    EthernetClient client_;
    IPAddress svr_ip_;
    uint16_t svr_port_;

    TaskPayload current_task_;
    ResultPayload current_result_;
    
    bool is_connected_;

    static uint8_t input_buffer_[64 * 1024];
    static uint8_t output_buffer_[64 * 1024];
};

#endif // WORKER_H