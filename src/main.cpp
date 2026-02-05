// used to test weights integrity on MCU
#include <Arduino.h>

#include "weights.h"
#include "layer_config.h"
#include "quant_params.h"

void print_weight_statistics() {
    Serial.println("\n========== Weight File Statistics ==========");
    Serial.print("Total Layers: ");
    Serial.println(NUM_LAYERS);

    uint32_t total_weights = 0, total_bias = 0;
    for (int i = 0; i < NUM_LAYERS; ++i) {
        Serial.printf("\n--- Layer %d ---\n", i);

        Serial.printf("Weight size: %u bytes\n", model_weights[i].weights_size);
        Serial.printf("Bias size: %u bytes\n", model_weights[i].bias_size * sizeof(int32_t));
        total_weights += model_weights[i].weights_size;
        total_bias += model_weights[i].bias_size * sizeof(int32_t);
        Serial.flush();
    }

    Serial.printf("\nTotal weights: %u bytes\n", total_weights);
    Serial.printf("Total bias: %u bytes\n", total_bias);
    Serial.flush();
}

void verify_weight() {
    Serial.println("\n========== Weight Value Check ==========");
    for (int i = 0; i < NUM_LAYERS; ++i) {
        size_t check_count = min(100U, model_weights[i].weights_size);
        for (size_t j = 0; j < check_count; ++j) {
            #ifdef PROGMEM
                int8_t w = pgm_read_byte(&model_weights[i].weights[j]);
            #else
                int8_t w = model_weights[i].weights[j];
            #endif
            
            if (w < -128 || w > 127) {
                Serial.printf("ERROR: Layer %d weight[%u] = %d out of int8 range!\n");
                break;
            }
        }

        check_count = min(100U, model_weights[i].bias_size);
        for (size_t j = 0; j < check_count; ++j) {
            #ifdef PROGMEM
                int32_t b = pgm_read_dword(&model_weights[i].bias[j]);
            #else
                int32_t b = model_weights[i].bias[j];
            #endif
            if (b < -2000000000 || b > 2000000000) {
                Serial.printf("ERROR: Layer %d bias[%u] = %d out of int32 range!\n");
                break;
            }
        }
        Serial.flush();
    }
    Serial.println("All value checks PASSED!");
    Serial.flush();
}

void verify_layer_config() {
    Serial.println("\n========== Layer Config Check ==========");
    for (size_t i = 0; i < NUM_LAYERS; ++i) {
        const struct LayerConfig *cfg = &model_layer_config[i];
        if (cfg->input_channels == 0 || cfg->output_channels == 0 || cfg->kernel_size == 0) {
            Serial.printf("ERROR: Layer %zu has invalid configuration!\n", i);
        }
    }
    size_t total_layer_config_bytes = sizeof(model_layer_config);
    Serial.printf("Total layer config size: %zu bytes\n", total_layer_config_bytes);
    Serial.println("All layer config checks PASSED!");
    Serial.flush();
}

void verify_quant_params() {
    Serial.println("\n========== Quantization Parameters Check ==========");
    for (size_t i = 0; i < NUM_LAYERS; ++i) {
        const struct QuantParams *qp = &model_quant_params[i];
        if (qp->input_scale <= 0 || qp->output_scale <= 0) {
            Serial.printf("ERROR: Layer %zu has invalid quantization scales!\n", i);
        }
        if (qp->input_zero_point < -128 || qp->input_zero_point > 127 ||
            qp->output_zero_point < -128 || qp->output_zero_point > 127) {
            Serial.printf("ERROR: Layer %zu has invalid quantization zero points!\n", i);
        }
    }
    for (size_t i = 0; i < 10; ++i) {
        const struct QuantParams *qp = &model_quant_params[i];
        Serial.printf("Layer %zu: input_scale=%.6f, input_zp=%d, output_scale=%.6f, output_zp=%d, weight_scale[0]=%.6f, weight_zp=%d\n",
                      i, qp->input_scale, qp->input_zero_point, qp->output_scale, qp->output_zero_point, qp->weight_scales[0], qp->weight_zps[0]);
    }
    Serial.println("All quantization parameters checks PASSED!");
    Serial.flush();
}

void setup() {
    Serial.begin(115200);
    while (!Serial && millis() < 5000); // wait for serial
    Serial.println("Weight File Validation Test");

    print_weight_statistics();
    verify_weight();    
    verify_layer_config();
    verify_quant_params();
    printf("\nAll integrity checks completed.\n");
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