#ifndef LINEAR_H
#define LINEAR_H

#include <Arduino.h>
#include <stdint.h>

struct LayerConfig; // TODO Need to rethink where should we put the struct
struct QuantParams; // TODO Need to rethink where should we put the struct

namespace linear {
    
    void native_linear(const uint8_t *input, const int8_t *weights, const int32_t *bias, 
                        uint8_t *output, const LayerConfig *cfg, const QuantParams *qp);

    // Slower than native one
    void dsp_linear(const uint8_t *input, const int8_t *weights, const int32_t *bias, 
                        uint8_t *output, const LayerConfig *cfg, const QuantParams *qp);
}


#endif