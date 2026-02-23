#ifndef RANS_H
#define RANS_H

#include <cstdint>
#include <cstddef>
#include <cstring>


namespace rans {

static constexpr uint32_t NUM_SYMBOLS = 256; // for byte data
static constexpr uint32_t PROB_BITS = 12; // number of bits for probabilities
static constexpr uint32_t PROB_SCALE = 1u << PROB_BITS; // total frequency count (4096)
static constexpr uint32_t RANS_L = 1u << 23; // renormalization threshold (must be > PROB_SCALE)
static constexpr uint32_t MAGIC = 0x72414E53;

struct Header {
    uint32_t magic;
    uint32_t original_size;
    uint32_t compressed_size;
    uint16_t freq[NUM_SYMBOLS]; // normalized frequencies (uint16_t to save space)
} __attribute__((packed));

// Per-symbol encoding parameters — indexed by symbol (0–255).
// Packs freq, cum, reciprocal, and renorm threshold into one struct
// so the hot loop only needs one table lookup per byte.
// 12 bytes × 256 = 3072 bytes on stack (fits comfortably).
struct EncSymbol {
    uint16_t freq;     // normalized frequency
    uint16_t cum;      // cumulative frequency (start position in [0, PROB_SCALE))
    uint32_t rcp;      // floor(2^32 / freq) for freq >= 2; unused when freq < 2
    uint32_t x_max;    // renormalization threshold
};

inline void count_frequencies(const uint8_t *data, size_t len, uint32_t out[NUM_SYMBOLS]) {
    std::memset(out, 0, NUM_SYMBOLS * sizeof(uint32_t));
    for (size_t i = 0; i < len; ++i) {
        ++out[data[i]];
    }
}

/// Scale raw counts so Σfreq == PROB_SCALE, keeping freq[s] >= 1
/// for every symbol s that appeared at least once.
inline void normalize(const uint32_t raw[NUM_SYMBOLS], uint16_t norm[NUM_SYMBOLS]) {
    uint32_t total = 0;
    for (uint32_t i = 0; i < NUM_SYMBOLS; ++i) {
        total += raw[i];
    }
    if (0 == total) {
        std::memset(norm, 0, NUM_SYMBOLS * sizeof(uint16_t));
        return;
    }
    
    uint32_t assigned = 0;
    for (uint32_t i = 0; i < NUM_SYMBOLS; ++i) {
        if (0 == raw[i]) {
            norm[i] = 0;
            continue;
        }
        norm[i] = static_cast<uint16_t>((static_cast<uint64_t>(raw[i]) * PROB_SCALE) / total);
        if (0 == norm[i]) {
            norm[i] = 1;
        }
        assigned += norm[i];
    }
    // correct rounding residual on the most-frequent symbol so Σfreq == PROB_SCALE
    int32_t diff = static_cast<int32_t>(PROB_SCALE) - static_cast<int32_t>(assigned);
    if (diff != 0) {
        uint32_t best = 0;
        for (uint32_t i = 1; i < NUM_SYMBOLS; ++i)
            if (raw[i] > raw[best]) best = i;
        norm[best] = static_cast<uint16_t>(
            static_cast<int32_t>(norm[best]) + diff);
    }
}

inline void build_cum(const uint16_t freq[NUM_SYMBOLS], uint16_t cum[NUM_SYMBOLS + 1]) {
    cum[0] = 0;
    for (uint32_t i = 0; i < NUM_SYMBOLS; ++i) {
        cum[i + 1] = cum[i] + freq[i];
    }
}

inline void build_enc_symbols(const uint16_t freq[NUM_SYMBOLS],
                              const uint16_t cum[NUM_SYMBOLS + 1],
                              EncSymbol syms[NUM_SYMBOLS]) {
    for (uint32_t s = 0; s < NUM_SYMBOLS; ++s) {
        syms[s].freq = freq[s];
        syms[s].cum  = cum[s];
        if (freq[s] >= 2) {
            // floor(2^32 / f) — always fits in uint32_t for f >= 2
            syms[s].rcp = static_cast<uint32_t>((1ULL << 32) / freq[s]);
        } else {
            syms[s].rcp = 0; // unused for f=0 or f=1
        }
        syms[s].x_max = freq[s] > 0
            ? ((RANS_L >> PROB_BITS) << 8) * freq[s]
            : 0;
    }
}
    
// compress
// return total bytes or 0 on failure
size_t compress(const uint8_t *input, size_t input_size, uint8_t *output, size_t output_capacity) {
    if (input_size == 0 || output_capacity == 0) {
        return 0;
    }

    constexpr size_t HEADER_SIZE = sizeof(Header);

    // 1. build frequency table
    uint32_t raw[NUM_SYMBOLS];
    count_frequencies(input, input_size, raw);

    Header *hdr = reinterpret_cast<Header *>(output);
    normalize(raw, hdr->freq);

    uint16_t cum[NUM_SYMBOLS + 1];
    build_cum(hdr->freq, cum);

    // 2. build per-symbol encoding table (256 entries, 3KB)
    EncSymbol syms[NUM_SYMBOLS];
    build_enc_symbols(hdr->freq, cum, syms);

    // 3. rANS encode (scan input backwards, write payload backwards)
    uint8_t * const payload_start = output + HEADER_SIZE;
    uint8_t *ptr = output + output_capacity;
    
    uint32_t x = RANS_L; // initial state
    for (size_t i = input_size; i-- > 0; ) {
        const EncSymbol &es = syms[input[i]];

        // renormalize: emit low bytes while state is too large for this symbol
        while (x >= es.x_max) {
            if (ptr <= payload_start) {
                return 0;
            }
            *--ptr = static_cast<uint8_t>(x);
            x >>= 8;
        }

        // rANS encode: C(x, s) = floor(x / f) * PROB_SCALE + (x mod f) + cum
        if (es.freq == 1) {
            // f=1: x/1 = x, x%1 = 0 — no division needed
            x = x * PROB_SCALE + es.cum;
        } else {
            // Reciprocal multiplication: q = floor((x * floor(2^32/f)) / 2^32)
            // This UNDERESTIMATES floor(x/f) by at most 1.
            // Proof: floor(2^32/f) >= (2^32-f)/f, so q >= x*(2^32-f)/(f*2^32)
            //        = x/f - x/2^32. Since x < 2^31, error < 0.5 < 1.
            uint32_t q = static_cast<uint32_t>((uint64_t(x) * es.rcp) >> 32);
            uint32_t r = x - q * es.freq;
            // Correction: if q was 1 too low, r >= f. At most one correction needed.
            if (r >= es.freq) { ++q; r -= es.freq; }
            x = q * PROB_SCALE + r + es.cum;
        }
    }

    // flush final state — 4 bytes, big-endian
    if (ptr - payload_start < 4) return 0;
    *--ptr = static_cast<uint8_t>(x >>  0);
    *--ptr = static_cast<uint8_t>(x >>  8);
    *--ptr = static_cast<uint8_t>(x >> 16);
    *--ptr = static_cast<uint8_t>(x >> 24);

    // move payload to right after header
    const size_t payload_size = static_cast<size_t>((output + output_capacity) - ptr);
    std::memmove(payload_start, ptr, payload_size);

    hdr->magic = MAGIC;
    hdr->original_size = static_cast<uint32_t>(input_size);
    hdr->compressed_size = static_cast<uint32_t>(payload_size);
    return HEADER_SIZE + payload_size;
}

// decompress
// Returns decompressed size or 0 on failure
inline size_t decompress(const uint8_t *input, size_t input_size, uint8_t *output, size_t output_capacity) {
    if (input_size < sizeof(Header)) return 0;

    const Header *hdr = reinterpret_cast<const Header *>(input);
    if (hdr->magic != MAGIC) return 0;
    if (hdr->original_size > output_capacity) return 0;

    const uint32_t orig_size = hdr->original_size;
    const uint32_t comp_size = hdr->compressed_size;

    if (input_size < sizeof(Header) + comp_size) return 0;

    // Build cumulative frequency table
    uint16_t cum[NUM_SYMBOLS + 1];
    build_cum(hdr->freq, cum);

    // Build reverse LUT: cum_value → symbol (4 KB on stack)
    uint8_t lut[PROB_SCALE];
    for (uint32_t s = 0; s < NUM_SYMBOLS; ++s) {
        for (uint16_t j = cum[s]; j < cum[s + 1]; ++j) {
            lut[j] = static_cast<uint8_t>(s);
        }
    }

    // Payload pointer
    const uint8_t *p   = input + sizeof(Header);
    const uint8_t *end = p + comp_size;
    if (end - p < 4) return 0;

    // Read initial state (big-endian 4 bytes)
    uint32_t state = (uint32_t(p[0]) << 24) | (uint32_t(p[1]) << 16)
                   | (uint32_t(p[2]) <<  8) |  uint32_t(p[3]);
    p += 4;

    const uint32_t mask = PROB_SCALE - 1;

    for (uint32_t i = 0; i < orig_size; ++i) {
        uint32_t c_val = state & mask;
        uint8_t  sym   = lut[c_val];
        uint16_t f     = hdr->freq[sym];
        uint16_t c     = cum[sym];

        state = f * (state >> PROB_BITS) + c_val - c;

        while (state < RANS_L && p < end) {
            state = (state << 8) | *p++;
        }

        output[i] = sym;
    }

    return orig_size;
}

} // namespace rans


#endif // RANS_H