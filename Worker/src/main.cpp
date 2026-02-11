#include <Arduino.h>
#include "worker.h"

#define WORKER_ID 0
#define SVR_IP IPAddress(192, 168, 1, 10)
#define SVR_PORT 54321

Worker worker(WORKER_ID, SVR_IP, SVR_PORT);

void setup() {
    Serial.begin(115200);
    while (!Serial);
    delay(1000);
    worker.Begin();
    Serial.println("Worker setup completed");
}

void loop() {
    worker.Loop();
    delay(10); // avoid busy loop, adjust as needed
    // static uint32_t last_heartbeat = 0;
    // if (millis() - last_heartbeat > 5000) {
    //     Serial.print(".");
    //     Serial.flush();
    //     last_heartbeat = millis();
    // }
}