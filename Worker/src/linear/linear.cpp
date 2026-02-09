#include "linear/linear.h"

#include <arm_math.h>
#include "weights.h"
#include "layer_config.h"
#include "quant_params.h"

namespace linear {

void native_linear(const uint8_t *input, const int8_t *weights, const int32_t *bias, 
                        uint8_t *output, const LayerConfig *cfg, const QuantParams *qp) {
    const uint32_t input_channels = cfg->input_channels;
    const uint32_t output_channels = qp->num_channels; // use this num because it's distributed
    
    for (size_t oc = 0; oc < output_channels; ++oc) {
        int32_t acc = bias[oc];
        float weight_scale = qp->weight_scales[oc];
        int32_t weight_zp = qp->weight_zps[oc];
        float multiplier = (qp->input_scale * weight_scale) / qp->output_scale;

        // weights @ input
        for (size_t ic = 0; ic < input_channels; ++ic) {
            int32_t input_val = (int32_t)input[ic] - qp->input_zero_point;
            int32_t weight_val = (int32_t)weights[oc * input_channels + ic] - weight_zp;
            acc += input_val * weight_val;
        }

        // requantize
        float acc_float = acc * multiplier + qp->output_zero_point;
        output[oc] = (uint8_t) max( 0, min( 255, (int32_t)roundf(acc_float)) );
    }
}

// TODO: It's a notice here. weight_buffer is too big to fit in with the 512KB RAM,
// so we need to reuse the buffer for each output channel. But now the speed is even slower than native one
// because of the overhead of copying weights for each output channel. 
// We need to find a better way to do this, maybe we can copy weights for multiple output channels at once and compute them together to amortize the overhead.
void _dsp_linear(const uint8_t *input, const int8_t *weights, const int32_t *bias, 
                        uint8_t *output, const LayerConfig *cfg, const QuantParams *qp,
                        int16_t *input_buffer, int16_t *weight_buffer) {
    const uint32_t input_channels = cfg->input_channels;
    const uint32_t output_channels = qp->num_channels; // use this num because it's distributed

    for (size_t i = 0; i < input_channels; ++i) {
        input_buffer[i] = (int16_t)input[i] - qp->input_zero_point;
    }

    for (size_t oc = 0; oc < output_channels; ++oc) {
        for (size_t ic = 0; ic < input_channels; ++ic) {
            weight_buffer[ic] = (int16_t)weights[oc * input_channels + ic] - qp->weight_zps[oc];
        }

        int64_t acc_q63 = 0;
        float multiplier = (qp->input_scale * qp->weight_scales[oc]) / qp->output_scale;
        arm_dot_prod_q15(weight_buffer, input_buffer, input_channels, &acc_q63);
        int32_t acc = (int32_t)acc_q63 + bias[oc];
        float acc_float = acc * multiplier + qp->output_zero_point;
        output[oc] = (uint8_t) max( 0, min(255, (int32_t) roundf(acc_float) ) );
    }
}

void dsp_linear(const uint8_t *input, const int8_t *weights, const int32_t *bias, 
                        uint8_t *output, const LayerConfig *cfg, const QuantParams *qp) {                            
    const uint32_t input_channels = cfg->input_channels;
    const uint32_t output_channels = qp->num_channels; // use this num because it's distributed

    std::vector<int16_t> weight_buffer(input_channels);
    std::vector<int16_t> input_buffer(input_channels);

    _dsp_linear(input, weights, bias, output, cfg, qp, input_buffer.data(), weight_buffer.data());
}



    // Serial.println("Hey, I'm in dsp_linear!!");
    // std::vector<int16_t> weight_buffer(output_channels * input_channels);
    // std::vector<int16_t> input_buffer(input_channels); // weight_buffer x input_buffer -> output_buffer
    // Serial.println("Hey, I'm in dsp_linear!");
    // for (size_t i = 0; i < input_channels; ++i) {
    //     input_buffer[i] = (int16_t)input[i] - qp->input_zero_point;
    // }
    // Serial.println("Hey, I'm in dsp_linear!");
    // for (size_t oc = 0; oc < output_channels; ++oc) {
    //     for (size_t ic = 0; ic < input_channels; ++ic) {
    //         weight_buffer[oc *input_channels + ic] = (int16_t)weights[oc * input_channels + ic] - qp->weight_zps[oc];
    //     }
    // }

    // Serial.println("Hey, I'm in dsp_linear!");
    
    // // compute with dsp
    // for (size_t oc = 0; oc < output_channels; ++oc) {
    //     int64_t acc_q63 = 0;
    //     float multiplier = (qp->input_scale * qp->weight_scales[oc]) / qp->output_scale;
    //     arm_dot_prod_q15(weight_buffer.data() + oc * input_channels, input_buffer.data(), input_channels, &acc_q63);
    //     int32_t acc = (int32_t)acc_q63 + bias[oc];
    //     float acc_float = acc * multiplier + qp->output_zero_point;
    //     output[oc] = (uint8_t) max( 0, min(255, (int32_t) roundf(acc_float) ) );
    // }



} // namespace linear