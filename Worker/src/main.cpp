// #include <Arduino.h>
// #include "conv/conv2d.h"


// void setup() {
//   Serial.begin(115200);
//   while (!Serial);
//   delay(1000);
//   conv2d::native_conv2d(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 4, 4); // just to test the function call
//   Serial.println("Conv Layer Test");
//   Serial.flush();
// }

// void loop() {
//   delay(1000);
//   static uint32_t last_heartbeat = 0;
//   if (millis() - last_heartbeat > 5000) {
//       Serial.print(".");
//       Serial.flush();
//       last_heartbeat = millis();
//   }
// }