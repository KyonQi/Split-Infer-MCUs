/*
 * rans_decode.c — Fast rANS byte decoder callable from Python via ctypes.
 * Matches the Teensy C++ encoder protocol exactly.
 *
 * Compile:
 *   gcc -O3 -shared -fPIC -o librans_decode.so rans_decode.c
 */

#include <stdint.h>
#include <stddef.h>
#include <string.h>

#define NUM_SYMS   256
#define PROB_BITS  12
#define PROB_SCALE (1u << PROB_BITS)   /* 4096 */
#define RANS_L     (1u << 23)
#define MAGIC      0x72414E53u

/* Must match the Teensy Header struct (little-endian, packed) */
typedef struct __attribute__((packed)) {
    uint32_t magic;
    uint32_t original_size;
    uint32_t compressed_size;
    uint16_t freq[NUM_SYMS];
} RansHeader;

/*
 * rans_decompress_c
 *
 * src      : compressed stream (header + payload)
 * src_len  : total bytes in src
 * dst      : output buffer (caller-allocated, >= original_size)
 * dst_cap  : capacity of dst buffer
 *
 * Returns  : original_size on success, 0 on failure.
 */
uint32_t rans_decompress_c(const uint8_t *src, uint32_t src_len,
                           uint8_t *dst, uint32_t dst_cap)
{
    if (src_len < sizeof(RansHeader))
        return 0;

    const RansHeader *hdr = (const RansHeader *)src;
    if (hdr->magic != MAGIC)
        return 0;
    if (hdr->original_size > dst_cap)
        return 0;

    const uint32_t orig_size = hdr->original_size;
    const uint32_t comp_size = hdr->compressed_size;

    if (src_len < sizeof(RansHeader) + comp_size)
        return 0;

    /* Build cumulative frequency table */
    uint16_t cum[NUM_SYMS + 1];
    cum[0] = 0;
    for (int i = 0; i < NUM_SYMS; ++i)
        cum[i + 1] = cum[i] + hdr->freq[i];

    /* Build reverse LUT: cum_value → symbol (4 KB on stack) */
    uint8_t lut[PROB_SCALE];
    for (int s = 0; s < NUM_SYMS; ++s)
        for (uint16_t j = cum[s]; j < cum[s + 1]; ++j)
            lut[j] = (uint8_t)s;

    /* Payload pointer */
    const uint8_t *p   = src + sizeof(RansHeader);
    const uint8_t *end = p + comp_size;
    if (end - p < 4)
        return 0;

    /* Read initial state (big-endian 4 bytes) */
    uint32_t state = ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16)
                   | ((uint32_t)p[2] <<  8) |  (uint32_t)p[3];
    p += 4;

    const uint32_t mask = PROB_SCALE - 1;

    for (uint32_t i = 0; i < orig_size; ++i) {
        uint32_t c_val = state & mask;
        uint8_t  sym   = lut[c_val];
        uint16_t f     = hdr->freq[sym];
        uint16_t c     = cum[sym];

        state = f * (state >> PROB_BITS) + c_val - c;

        while (state < RANS_L && p < end)
            state = (state << 8) | *p++;

        dst[i] = sym;
    }

    return orig_size;
}
