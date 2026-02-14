#include <Arduino.h>
#include <arm_math.h>
#include <memory>

#include "linear/linear.h"
#include "linear/linear_test_data.h"
#include "weights.h"
#include "layer_config.h"
#include "quant_params.h"

void test_single_linear_layer() {
    Serial.println("\n========== Single Linear Layer Test ==========");

    const int8_t *weights = model_weights[52].weights;
    const int32_t *bias = model_weights[52].bias;
    const LayerConfig *cfg = &model_layer_config[52];
    const QuantParams *qp = &model_quant_params[52];

    // output buffer
    uint8_t output[qp->num_channels];

    uint32_t start = micros();
    linear::native_linear(&test_input[0], weights, bias, &output[0], cfg, qp);
    uint32_t elapsed = micros() - start;
    
    Serial.printf("Input: 1280, Output: 250\n");
    Serial.printf("LINEAR: Inference time: %u us\n", elapsed);
    // Serial.println("Output:");
    // for (size_t c = 0; c < qp->num_channels; ++c) {
    //     Serial.printf("%u ", output[c]);
    // }
    // Serial.println("\n============================================");
}

void setup() {
    Serial.begin(115200);
    while (!Serial);
    delay(1000);
    Serial.println("Linear Layer Test");
    test_single_linear_layer();
    Serial.flush();
}

void loop() {
    delay(1000);
    static uint32_t last_heartbeat = 0;
    if (millis() - last_heartbeat > 5000) {
        Serial.print(".");
        Serial.flush();
        last_heartbeat = millis();
    }
}