#include "conv/conv2d.h"

#include <arm_math.h>
#include <memory>
#include <assert.h>

#include "weights.h"
#include "layer_config.h"
#include "quant_params.h"

namespace conv2d {
    
void native_conv2d(const uint8_t *input, const int8_t *weights, const int32_t *bias, 
                    uint8_t *output, const LayerConfig *cfg, const QuantParams *qp,
                    const uint8_t in_h, const uint8_t in_w) {
    // native convolution implementation for testing
    // const int in_h = 4, in_w = 4;
    const int out_h = (in_h + 2 * cfg->padding - cfg->kernel_size) / cfg->stride + 1;
    const int out_w = (in_w + 2 * cfg->padding - cfg->kernel_size) / cfg->stride + 1;

    for (size_t oc = 0; oc < cfg->output_channels; ++oc) {
        int32_t bias_val = bias[oc];
        float weight_scale = qp->weight_scales[oc];
        int weight_zero_point = qp->weight_zps[oc]; // must be 0
        float multiplier = (qp->input_scale * weight_scale) / qp->output_scale;

        for (size_t oh = 0; oh < out_h; ++oh) {
            for (size_t ow = 0; ow < out_w; ++ow) {
                int32_t acc = bias_val;

                int start_y = oh * cfg->stride - cfg->padding; // input y coordinate corresponding to output (oh, ow)
                int start_x = ow * cfg->stride - cfg->padding; // input x coordinate corresponding to output (oh, ow)

                for (size_t ic = 0; ic < cfg->input_channels; ++ic) {
                    for (size_t kh = 0; kh < cfg->kernel_size; ++kh) {
                        for (size_t kw = 0; kw < cfg->kernel_size; ++kw) {
                            int in_y = start_y + kh;
                            int in_x = start_x + kw;

                            // Check for valid input coordinates (handle padding)
                            if (in_y >= 0 && in_y < in_h && in_x >= 0 && in_x < in_w) {
                                int32_t input_val = (int32_t) input[ic * in_h * in_w + in_y * in_w + in_x] - qp->input_zero_point;
                                int32_t weight_val = (int32_t) weights[oc * cfg->input_channels * cfg->kernel_size * cfg->kernel_size +
                                                            ic * cfg->kernel_size * cfg->kernel_size +
                                                            kh * cfg->kernel_size + kw] - weight_zero_point;
                                acc += input_val * weight_val;
                            }
                        }
                    }
                }

                // requantize
                float acc_float = acc * multiplier + qp->output_zero_point;
                int o_idx = oc * out_h * out_w + oh * out_w + ow;
                output[o_idx] = (uint8_t) max( 0, min(255, (int32_t) roundf(acc_float) ) );      
            }
        }

    }
}

// Im2Col + GeMM implementations
// 1. input -> im2col buffer : [in_c, in_h, in_w] -> [in_c * kernel_h * kernel_w, out_h * out_w]
// 2. weight -> weight buffer : [out_c, in_c, kernel_h, kernel_w] -> [out_c, in_c * kernel_h * kernel_w]
// 3. GeMM : weight_buffer @ im2col_buffer + bias -> output_buffer 
// 4. requantize and transform output
void _im2col_conv2d(const uint8_t *input, std::vector<q15_t> &col_buffer, const LayerConfig *cfg, const QuantParams *qp,
                    const uint8_t in_h, const uint8_t in_w) {
    // const int in_h = 4, in_w = 4;
    const int out_h = (in_h + 2 * cfg->padding - cfg->kernel_size) / cfg->stride + 1;
    const int out_w = (in_w + 2 * cfg->padding - cfg->kernel_size) / cfg->stride + 1;

    const int col_rows = cfg->input_channels * cfg->kernel_size * cfg->kernel_size;
    const int col_cols = out_h * out_w;

    int col_idx = 0;

    // since we are using dot-product, we need to transpose the im2col output to [out_h * out_w, in_c * kernel_h * kernel_w]
    for (size_t oh = 0; oh < out_h; ++oh) {
        for (size_t ow = 0; ow < out_w; ++ow) {
            for (size_t ic = 0; ic < cfg->input_channels; ++ic) {
                for (size_t kh = 0; kh < cfg->kernel_size; ++kh) {
                    for (size_t kw = 0; kw < cfg->kernel_size; ++kw) {
                        int in_y = oh * cfg->stride - cfg->padding + kh;
                        int in_x = ow * cfg->stride - cfg->padding + kw;
                        int32_t input_val = 0;
                        if (in_y >= 0 && in_y < in_h && in_x >= 0 && in_x < in_w) {
                            input_val = (int32_t) input[ic * in_h * in_w + in_y * in_w + in_x] - qp->input_zero_point;
                        }
                        col_buffer[col_idx++] = (q15_t) input_val;
                    }
                }
            }
        }
    }
}

void _prepare_weights(const int8_t *weights, std::vector<q15_t> &weight_buffer, const LayerConfig *cfg, const QuantParams *qp) {
    // weights [out_c, in_c, kh, kw] -> [out_c, in_c * kh * kw]
    // TODO maybe we can use mem replacement for better mem usage
    const int col_rows = cfg->input_channels * cfg->kernel_size * cfg->kernel_size;
    int weight_idx = 0;

    for (size_t oc = 0; oc < cfg->output_channels; ++oc) {
        int32_t weight_zero_point = qp->weight_zps[oc];
        for (size_t i = 0; i < col_rows; ++i) {
            weight_buffer[weight_idx++] = (q15_t) (weights[oc * col_rows + i] - weight_zero_point);
        }
    }
}

void _gemm(q15_t *col_buffer, q15_t *weight_buffer,
            const int32_t *bias, uint8_t *output, const LayerConfig *cfg, const QuantParams *qp,
            const uint8_t in_h, const uint8_t in_w) {
    // const int in_h = 4, in_w = 4;
    const int out_h = (in_h + 2 * cfg->padding - cfg->kernel_size) / cfg->stride + 1;
    const int out_w = (in_w + 2 * cfg->padding - cfg->kernel_size) / cfg->stride + 1;

    const int col_rows = cfg->input_channels * cfg->kernel_size * cfg->kernel_size;
    const int col_cols = out_h * out_w;

    int out_idx = 0;

    for (size_t oc = 0; oc < cfg->output_channels; ++oc) {
        int64_t acc_q63 = 0;
        float multiplier = (qp->input_scale * qp->weight_scales[oc]) / qp->output_scale;

        for (size_t p = 0; p < col_cols; ++p) {
            arm_dot_prod_q15(weight_buffer + oc * col_rows, col_buffer + p * col_rows, col_rows, &acc_q63);

            int32_t acc = (int32_t) acc_q63 + bias[oc];
            
            float acc_float = acc * multiplier + qp->output_zero_point;
            output[out_idx++] = (uint8_t) max( 0, min(255, (int32_t) roundf(acc_float) ) );
        }
    }

}

void im2col_conv2d(const uint8_t *input, const int8_t *weights, const int32_t *bias, 
                    uint8_t *output, const LayerConfig *cfg, const QuantParams *qp,
                    const uint8_t in_h, const uint8_t in_w) {
    // const int in_h = 4, in_w = 4;
    const int out_h = (in_h + 2 * cfg->padding - cfg->kernel_size) / cfg->stride + 1;
    const int out_w = (in_w + 2 * cfg->padding - cfg->kernel_size) / cfg->stride + 1;

    const int col_rows = cfg->input_channels * cfg->kernel_size * cfg->kernel_size;
    const int col_cols = out_h * out_w;
    
    // allocate buffers TODO maybe static allocation
    std::vector<q15_t> col_buffer(col_cols * col_rows); // it's a transpose of ideal im2col output
    std::vector<q15_t> weight_buffer(cfg->output_channels * col_rows); // TODO We can use a conv_work_space struct to reuse the buffer

    // 1. im2col
    _im2col_conv2d(input, col_buffer, cfg, qp, in_h, in_w);
    // 2. transform weights
    _prepare_weights(weights, weight_buffer, cfg, qp);
    // 3. GeMM with DSP
    _gemm(col_buffer.data(), weight_buffer.data(), bias, output, cfg, qp, in_h, in_w);
}


// depthwise conv
void depthwise_conv2d(const uint8_t *input, const int8_t *weights, const int32_t *bias, 
                    uint8_t *output, const LayerConfig *cfg, const QuantParams *qp, 
                    const uint8_t in_h, const uint8_t in_w) {
    assert(cfg->input_channels == cfg->output_channels);

    const int out_h = (in_h + 2 * cfg->padding - cfg->kernel_size) / cfg->stride + 1;
    const int out_w = (in_w + 2 * cfg->padding - cfg->kernel_size) / cfg->stride + 1;

    for (size_t oc = 0; oc < cfg->output_channels; ++oc) {
        int32_t bias_val = bias[oc];
        float weight_scale = qp->weight_scales[oc];
        int weight_zero_point = qp->weight_zps[oc]; // must be 0
        float multiplier = (qp->input_scale * weight_scale) / qp->output_scale;

        for (size_t oh = 0; oh < out_h; ++oh) {
            for (size_t ow = 0; ow < out_w; ++ow) {
                int32_t acc = bias_val;

                int start_y = oh * cfg->stride - cfg->padding; // input y coordinate corresponding to output (oh, ow)
                int start_x = ow * cfg->stride - cfg->padding; // input x coordinate corresponding to output (oh, ow)

                for (size_t kh = 0; kh < cfg->kernel_size; ++kh) {
                    for (size_t kw = 0; kw < cfg->kernel_size; ++kw) {
                        int in_y = start_y + kh;
                        int in_x = start_x + kw;
                        
                        // Check for valid input coordinates (handle padding)
                        if (in_y >= 0 && in_y < in_h && in_x >= 0 && in_x < in_w) {
                            int32_t input_val = (int32_t) input[oc * in_h * in_w + in_y * in_w + in_x] - qp->input_zero_point;
                            int32_t weight_val = (int32_t) weights[oc * cfg->kernel_size * cfg->kernel_size +
                                                        kh * cfg->kernel_size + kw] - weight_zero_point;
                            acc += input_val * weight_val;
                        }
                    }
                }

                // requantize
                float acc_float = acc * multiplier + qp->output_zero_point;
                int o_idx = oc * out_h * out_w + oh * out_w + ow;
                output[o_idx] = (uint8_t) max( 0, min(255, (int32_t) roundf(acc_float) ) );      
            }
        }

    }
}




    
} // namespace conv2d