#include "worker.h"
#include "protocol.h"

Worker::Worker(uint8_t worker_id, IPAddress svr_ip, uint16_t svr_port)
    : worker_id_(worker_id), svr_ip_(svr_ip), svr_port_(svr_port), is_connected_(false) {
}

Worker::~Worker() {}

void Worker::Begin() {
    // allocate static IP
    uint8_t mac[6] = { 0xDE, 0xAD, 0xBE, 0xEF, 0xFE, worker_id_ };
    IPAddress local_ip(192, 168, 1, 100 + worker_id_);
    IPAddress dns(192, 168, 1, 1);
    IPAddress gateway(192, 168, 1, 1);
    IPAddress subnet(255, 255, 255, 0);

    Ethernet.begin(mac, local_ip, dns, gateway, subnet); // TODO check here

    Serial.printf("Worker %d started with IP: %s\n", worker_id_, String(local_ip));

    ConnectToServer();
}

void Worker::ConnectToServer() {
    Serial.printf("Worker %d connecting to server %s:%d...\n", worker_id_, String(svr_ip_), svr_port_);

    while (0 == client_.connect(svr_ip_, svr_port_)) {
        Serial.printf("Worker %d failed to connect, retrying in 2s...\n", worker_id_);
        delay(2000);
    }

    Serial.printf("Connected to server %s:%d\n", String(svr_ip_), svr_port_);
    is_connected_ = true;

    SendRegistration();
}

void Worker::SendRegistration() {

}