// Unit tests for verifying model weights and biases retrieval and integrity
// It must run in a native C++ environment with Unity test framework ([env:native] in platformio.ini)
#include <unity.h>
#include <stdint.h>
#include <stdio.h>

#include "weights.h"

#ifdef NATIVE_TEST
    #define PROGMEM
    #define pgm_read_byte(addr) (*(const uint8_t *)(addr))
#endif

void setUp() {
}

void tearDown() {
}

void test_weights_not_null() {
    size_t num_layers = sizeof(model_weights) / sizeof(struct LayerWeights);
    for (size_t i = 0; i < num_layers; ++i) {
        TEST_ASSERT_NOT_NULL_MESSAGE(model_weights[i].weights, "Weights pointer is null");
        TEST_ASSERT_NOT_NULL_MESSAGE(model_weights[i].bias, "Bias pointer is null");
    }
}

void test_weights_readable() {
    const int8_t* first_weights = model_weights[0].weights;
    TEST_ASSERT_NOT_NULL_MESSAGE(first_weights, "First layer weights pointer is null");
    for (int i = 0; i < 10; ++i) {
        int8_t w = pgm_read_byte(&first_weights[i]);
        TEST_ASSERT_GREATER_THAN(-129, w);
        TEST_ASSERT_LESS_THAN(128, w);
        printf("Weight[%d]: %d\n", i, w);
    }
}

void test_bias_readable() {
    const int32_t *first_bias = model_weights[0].bias;
    TEST_ASSERT_NOT_NULL_MESSAGE(first_bias, "First layer bias pointer is null");
    for (int i = 0; i < 10; ++i) {
        int32_t b = first_bias[i];
        TEST_ASSERT_GREATER_THAN(-2000000000, b);
        TEST_ASSERT_LESS_THAN(2000000000, b);
        printf("Bias[%d]: %d\n", i, b);
    }
}

void print_memory_stats() {
    size_t num_layers = sizeof(model_weights) / sizeof(struct LayerWeights);
    printf("\n========== Memory Statistics ==========\n");
    printf("Total layers: %zu\n", num_layers);
    printf("LayerWeights struct size: %zu bytes\n", sizeof(struct LayerWeights));
    printf("Total array size: %zu bytes\n", sizeof(model_weights));
    printf("========================================\n\n");
}

int main() {
    UNITY_BEGIN();
    RUN_TEST(test_weights_not_null);
    RUN_TEST(test_weights_readable);
    RUN_TEST(test_bias_readable);
    print_memory_stats();
    return UNITY_END();
}