// Auto-generated layer configuration header file

#ifndef LAYER_CONFIG_H
#define LAYER_CONFIG_H

#include <stdint.h>

struct LayerConfig {
    const char* name;
    uint32_t input_channels;
    uint32_t output_channels;
    uint32_t kernel_size;
    uint32_t stride;
    uint32_t padding;
};

const struct LayerConfig model_layer_config[] = {
    {"init_conv", 3, 32, 3, 2, 1},
    {"blk0_dw", 32, 32, 3, 1, 1},
    {"blk0_proj", 32, 16, 1, 1, 0},
    {"blk1_exp", 16, 96, 1, 1, 0},
    {"blk1_dw", 96, 96, 3, 2, 1},
    {"blk1_proj", 96, 24, 1, 1, 0},
    {"blk2_exp", 24, 144, 1, 1, 0},
    {"blk2_dw", 144, 144, 3, 1, 1},
    {"blk2_proj", 144, 24, 1, 1, 0},
    {"blk3_exp", 24, 144, 1, 1, 0},
    {"blk3_dw", 144, 144, 3, 2, 1},
    {"blk3_proj", 144, 32, 1, 1, 0},
    {"blk4_exp", 32, 192, 1, 1, 0},
    {"blk4_dw", 192, 192, 3, 1, 1},
    {"blk4_proj", 192, 32, 1, 1, 0},
    {"blk5_exp", 32, 192, 1, 1, 0},
    {"blk5_dw", 192, 192, 3, 1, 1},
    {"blk5_proj", 192, 32, 1, 1, 0},
    {"blk6_exp", 32, 192, 1, 1, 0},
    {"blk6_dw", 192, 192, 3, 2, 1},
    {"blk6_proj", 192, 64, 1, 1, 0},
    {"blk7_exp", 64, 384, 1, 1, 0},
    {"blk7_dw", 384, 384, 3, 1, 1},
    {"blk7_proj", 384, 64, 1, 1, 0},
    {"blk8_exp", 64, 384, 1, 1, 0},
    {"blk8_dw", 384, 384, 3, 1, 1},
    {"blk8_proj", 384, 64, 1, 1, 0},
    {"blk9_exp", 64, 384, 1, 1, 0},
    {"blk9_dw", 384, 384, 3, 1, 1},
    {"blk9_proj", 384, 64, 1, 1, 0},
    {"blk10_exp", 64, 384, 1, 1, 0},
    {"blk10_dw", 384, 384, 3, 1, 1},
    {"blk10_proj", 384, 96, 1, 1, 0},
    {"blk11_exp", 96, 576, 1, 1, 0},
    {"blk11_dw", 576, 576, 3, 1, 1},
    {"blk11_proj", 576, 96, 1, 1, 0},
    {"blk12_exp", 96, 576, 1, 1, 0},
    {"blk12_dw", 576, 576, 3, 1, 1},
    {"blk12_proj", 576, 96, 1, 1, 0},
    {"blk13_exp", 96, 576, 1, 1, 0},
    {"blk13_dw", 576, 576, 3, 2, 1},
    {"blk13_proj", 576, 160, 1, 1, 0},
    {"blk14_exp", 160, 960, 1, 1, 0},
    {"blk14_dw", 960, 960, 3, 1, 1},
    {"blk14_proj", 960, 160, 1, 1, 0},
    {"blk15_exp", 160, 960, 1, 1, 0},
    {"blk15_dw", 960, 960, 3, 1, 1},
    {"blk15_proj", 960, 160, 1, 1, 0},
    {"blk16_exp", 160, 960, 1, 1, 0},
    {"blk16_dw", 960, 960, 3, 1, 1},
    {"blk16_proj", 960, 320, 1, 1, 0},
    {"final_conv", 320, 1280, 1, 1, 0},
    {"fc_final", 1280, 1000, 1, 1, 0},
};

#endif // LAYER_CONFIG_H
