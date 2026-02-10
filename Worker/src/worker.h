#ifndef WORKER_H
#define WORKER_H

#include <Arduino.h>
#include <NativeEthernet.h>

class Worker final {
public:
    Worker(uint8_t worker_id, IPAddress svr_ip, uint16_t svr_port);

    ~Worker();

    void Begin();
    
private:
    void ConnectToServer();

    void SendRegistration();

    void ProcessTask();

    void SendResult();

private:
    uint8_t worker_id_;
    IPAddress svr_ip_;
    uint16_t svr_port_;
    
    EthernetClient client_;
    bool is_connected_;

    DMAMEM uint8_t input_buffer_[64 * 1024];
    DMAMEM uint8_t output_buffer_[64 * 1024];
};




#endif // WORKER_H