/*
 * rans_codec.c — Fast rANS byte encoder + decoder callable from Python via ctypes.
 * Matches the Teensy C++ codec protocol exactly.
 *
 * Compile:
 *   gcc -O3 -shared -fPIC -o src/librans_codec.so src/rans_codec.c
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

/* Per-symbol encoder parameters */
typedef struct {
    uint16_t freq;
    uint16_t cum;
    uint32_t rcp;    /* floor(2^32 / freq) for freq >= 2 */
    uint32_t x_max;  /* renormalization threshold */
} EncSymbol;

/* ──────────────────────────────────────────────────────────────────── */
/*                           ENCODER                                   */
/* ──────────────────────────────────────────────────────────────────── */

/*
 * rans_compress_c
 *
 * src      : raw input bytes
 * src_len  : number of input bytes
 * dst      : output buffer (caller-allocated, should be >= src_len + 524)
 * dst_cap  : capacity of dst buffer
 *
 * Returns  : total compressed size (header + payload) on success, 0 on failure.
 */
uint32_t rans_compress_c(const uint8_t *src, uint32_t src_len,
                         uint8_t *dst, uint32_t dst_cap)
{
    if (src_len == 0 || dst_cap < sizeof(RansHeader) + 4)
        return 0;

    /* 1. Count frequencies */
    uint32_t raw[NUM_SYMS];
    memset(raw, 0, sizeof(raw));
    for (uint32_t i = 0; i < src_len; ++i)
        ++raw[src[i]];

    /* 2. Normalize to PROB_SCALE */
    RansHeader *hdr = (RansHeader *)dst;
    uint32_t total = src_len;
    uint32_t assigned = 0;

    for (int i = 0; i < NUM_SYMS; ++i) {
        if (raw[i] == 0) {
            hdr->freq[i] = 0;
            continue;
        }
        hdr->freq[i] = (uint16_t)(((uint64_t)raw[i] * PROB_SCALE) / total);
        if (hdr->freq[i] == 0)
            hdr->freq[i] = 1;
        assigned += hdr->freq[i];
    }
    /* Correct rounding residual on most-frequent symbol */
    int32_t diff = (int32_t)PROB_SCALE - (int32_t)assigned;
    if (diff != 0) {
        uint32_t best = 0;
        for (uint32_t i = 1; i < NUM_SYMS; ++i)
            if (raw[i] > raw[best]) best = i;
        hdr->freq[best] = (uint16_t)((int32_t)hdr->freq[best] + diff);
    }

    /* 3. Build cumulative and encoder tables */
    uint16_t cum[NUM_SYMS + 1];
    cum[0] = 0;
    for (int i = 0; i < NUM_SYMS; ++i)
        cum[i + 1] = cum[i] + hdr->freq[i];

    EncSymbol syms[NUM_SYMS];
    for (int s = 0; s < NUM_SYMS; ++s) {
        syms[s].freq = hdr->freq[s];
        syms[s].cum  = cum[s];
        if (hdr->freq[s] >= 2)
            syms[s].rcp = (uint32_t)((1ULL << 32) / hdr->freq[s]);
        else
            syms[s].rcp = 0;
        syms[s].x_max = hdr->freq[s] > 0
            ? ((RANS_L >> PROB_BITS) << 8) * hdr->freq[s]
            : 0;
    }

    /* 4. Encode backwards */
    uint8_t *payload_start = dst + sizeof(RansHeader);
    uint8_t *ptr = dst + dst_cap;  /* write pointer, moves backwards */

    uint32_t x = RANS_L;  /* initial state */
    for (uint32_t i = src_len; i-- > 0; ) {
        const EncSymbol *es = &syms[src[i]];

        /* Renormalize */
        while (x >= es->x_max) {
            if (ptr <= payload_start)
                return 0;  /* output overflow */
            *--ptr = (uint8_t)x;
            x >>= 8;
        }

        /* rANS encode */
        if (es->freq == 1) {
            x = x * PROB_SCALE + es->cum;
        } else {
            uint32_t q = (uint32_t)(((uint64_t)x * es->rcp) >> 32);
            uint32_t r = x - q * es->freq;
            if (r >= es->freq) { ++q; r -= es->freq; }
            x = q * PROB_SCALE + r + es->cum;
        }
    }

    /* Flush final state (big-endian 4 bytes) */
    if (ptr - payload_start < 4)
        return 0;
    *--ptr = (uint8_t)(x >>  0);
    *--ptr = (uint8_t)(x >>  8);
    *--ptr = (uint8_t)(x >> 16);
    *--ptr = (uint8_t)(x >> 24);

    /* Move payload to right after header */
    uint32_t payload_size = (uint32_t)((dst + dst_cap) - ptr);
    memmove(payload_start, ptr, payload_size);

    hdr->magic           = MAGIC;
    hdr->original_size   = src_len;
    hdr->compressed_size = payload_size;

    return (uint32_t)sizeof(RansHeader) + payload_size;
}


/* ──────────────────────────────────────────────────────────────────── */
/*                           DECODER                                   */
/* ──────────────────────────────────────────────────────────────────── */

/*
 * rans_decompress_c  (kept for backward compatibility with existing Python code)
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

    /* Build reverse LUT */
    uint8_t lut[PROB_SCALE];
    for (int s = 0; s < NUM_SYMS; ++s)
        for (uint16_t j = cum[s]; j < cum[s + 1]; ++j)
            lut[j] = (uint8_t)s;

    const uint8_t *p   = src + sizeof(RansHeader);
    const uint8_t *end = p + comp_size;
    if (end - p < 4)
        return 0;

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
