#ifndef CONV2D_H
#define CONV2D_H

#include <Arduino.h>

struct LayerConfig; // TODO Need to rethink where should we put the struct
struct QuantParams; // TODO Need to rethink where should we put the struct

namespace conv2d {

    // normal conv
    void native_conv2d(const uint8_t *input, const int8_t *weights, const int32_t *bias, 
                        uint8_t *output, const LayerConfig *cfg, const QuantParams *qp, const uint8_t in_h, const uint8_t in_w);

    void native_conv2d_padded(const uint8_t *input, const int8_t *weights, const int32_t *bias, 
                    uint8_t *output, const LayerConfig *cfg, const QuantParams *qp,
                    const uint8_t in_h, const uint8_t in_w,
                    uint8_t pad_top, uint8_t pad_bottom, uint8_t pad_left, uint8_t pad_right);

    void im2col_conv2d(const uint8_t *input, const int8_t *weights, const int32_t *bias, 
                        uint8_t *output, const LayerConfig *cfg, const QuantParams *qp, const uint8_t in_h, const uint8_t in_w);

    // depthwise conv
    void depthwise_conv2d(const uint8_t *input, const int8_t *weights, const int32_t *bias, 
                        uint8_t *output, const LayerConfig *cfg, const QuantParams *qp, const uint8_t in_h, const uint8_t in_w);

    // depthwise conv with asymmetric padding (for block mode: worker-side DW padding after expand)
    void depthwise_conv2d_padded(const uint8_t *input, const int8_t *weights, const int32_t *bias,
                        uint8_t *output, const LayerConfig *cfg, const QuantParams *qp,
                        const uint8_t in_h, const uint8_t in_w,
                        uint8_t pad_top, uint8_t pad_bottom, uint8_t pad_left, uint8_t pad_right);

    // pointwise conv (1x1)
    void pointwise_conv2d(const uint8_t *input, const int8_t *weights, const int32_t *bias, 
                        uint8_t *output, const LayerConfig *cfg, const QuantParams *qp, const uint8_t in_h, const uint8_t in_w,
                        uint8_t *workspace, uint32_t workspace_size);

    // void depthwise_conv2d_dsp(const uint8_t *input, const int8_t *weights, const int32_t *bias, 
    //                     uint8_t *output, const LayerConfig *cfg, const QuantParams *qp, const uint8_t in_h, const uint8_t in_w);

} // namespace conv2d


#endif