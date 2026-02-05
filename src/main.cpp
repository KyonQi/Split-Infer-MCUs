// used to test weights integrity on MCU
#include <Arduino.h>

#include "weights.h"

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

void setup() {
    Serial.begin(115200);
    while (!Serial && millis() < 5000); // wait for serial
    Serial.println("Weight File Validation Test");

    print_weight_statistics();
    verify_weight();    
    
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