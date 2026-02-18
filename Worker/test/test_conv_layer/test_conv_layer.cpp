#include <Arduino.h>
#include <arm_math.h>
#include <memory>

#include "conv2d.h"
#include "conv2d_test_data.h"
#include "weights.h"
#include "layer_config.h"
#include "quant_params.h"

void test_single_conv_layer() {
    Serial.println("\n========== Single Conv Layer Test ==========");

    const int8_t *weights = model_weights[0].weights;
    const int32_t *bias = model_weights[0].bias;
    const LayerConfig *cfg = &model_layer_config[0];
    const QuantParams *qp = &model_quant_params[0];

    // output buffer
    uint8_t output[32][2][2];

    uint8_t output_im2col[32][2][2];

    uint32_t start = micros();
    conv2d::native_conv2d(&test_input[0][0][0], weights, bias, &output[0][0][0], cfg, qp, 4, 4);
    uint32_t elapsed = micros() - start;

    uint32_t start_im2col = micros();
    conv2d::im2col_conv2d(&test_input[0][0][0], weights, bias, &output_im2col[0][0][0], cfg, qp, 4, 4);
    uint32_t elapsed_im2col = micros() - start_im2col;

    Serial.printf("Input: 3x4x4\n");
    Serial.printf("CONV: Inference time: %u us\n", elapsed);
    Serial.printf("CONV: Inference time (im2col): %u us\n", elapsed_im2col);
    // Serial.println("Output:");
    // for (size_t c = 0; c < cfg->output_channels; ++c) {
    //     for (size_t h = 0; h < 2; ++h) {
    //         for (size_t w = 0; w < 2; ++w) {
    //             Serial.printf("%d ", output[c][h][w]);
    //         }
    //     }
    //     Serial.println();
    // }

    Serial.println("============================================");
    // for (size_t c = 0; c < cfg->output_channels; ++c) {
    //     for (size_t h = 0; h < 2; ++h) {
    //         for (size_t w = 0; w < 2; ++w) {
    //             Serial.printf("%d ", output_im2col[c][h][w]);
    //         }
    //     }
    //     Serial.println();
    // }
}

void test_depthwise_conv_layer() {
    Serial.println("\n========== Depthwise Conv Layer Test ==========");
    // choose 1st layer for testing blk0_dw (channel=32, stride=1, padding=1, kernel=3)
    const int8_t *weights = model_weights[1].weights;
    const int32_t *bias = model_weights[1].bias;
    const LayerConfig *cfg = &model_layer_config[1];
    const QuantParams *qp = &model_quant_params[1];

    // output buffer
    uint8_t output[32][4][4];
    uint32_t start = micros();
    conv2d::depthwise_conv2d(&test_input_dw[0][0][0], weights, bias, &output[0][0][0], cfg, qp, 4, 4);
    uint32_t elapsed = micros() - start;
    Serial.printf("Input: 32x4x4\n");
    Serial.printf("CONV: Inference time: %lu us\n", elapsed);

    Serial.println("============================================");
    // print channel 0 and 31
    // for (size_t h = 0; h < 4; ++h) {
    //     for (size_t w = 0; w < 4; ++w) {
    //         Serial.printf("%d ", output[0][h][w]);
    //     }
    // }
    // Serial.println();
    // Serial.println();
    // for (size_t h = 0; h < 4; ++h) {
    //     for (size_t w = 0; w < 4; ++w) {
    //         Serial.printf("%d ", output[31][h][w]);
    //     }
    // }
}

void setup() {
    Serial.begin(115200);
    while (!Serial);
    delay(1000);
    Serial.println("Conv Layer Test");
    Serial.flush();
    test_single_conv_layer();
    test_depthwise_conv_layer();
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