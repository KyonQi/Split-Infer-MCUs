#include "memory_tracker.h"

// ═══════════════════════════════════════════════════════════════════════
//  Teensy 4.1 Linker Symbols  (verified via arm-none-eabi-nm)
// ═══════════════════════════════════════════════════════════════════════
//
//  RAM1 (DTCM):
//    0x20062B00  _ebss       end of .bss (after input_buffer_ etc.)
//    0x20068000  _estack     top of DTCM / initial stack pointer
//       → free RAM1 ≈ SP - _ebss ≈ 21 KB (stack only, no heap in DTCM)
//
//  RAM2 (OCRAM2):
//    0x2025A8A0  _heap_start after DMAMEM vars (output_buffer_ + Ethernet)
//    0x20280000  _heap_end   top of OCRAM2
//       → free heap ≈ _heap_end - __brkval ≈ 150 KB
//
// ═══════════════════════════════════════════════════════════════════════

extern "C" {
    // RAM1 linker symbols
    extern unsigned long _ebss;    // end of .bss in DTCM
    extern unsigned long _estack;  // top of DTCM (= initial SP)

    // RAM2 linker symbols  (heap lives in OCRAM2)
    extern unsigned long _heap_start;
    extern unsigned long _heap_end;
    extern char *__brkval;         // current heap top (variable in DTCM, VALUE in RAM2)
}

namespace mem {

// ─── Low-Level RAM Query ─────────────────────────────────────────────

uint32_t freeRAM1() {
    // RAM1 (DTCM) layout:
    //   [.data][.bss][_ebss] ... FREE GAP (stack room) ... [SP↓ ... _estack]
    //
    // No heap in RAM1 on Teensy 4.1 — heap is in RAM2.
    // Free RAM1 = SP - _ebss = how much room the stack has before it
    // collides with .bss data.
    auto sp = (char *)__builtin_frame_address(0);
    return (uint32_t)(sp - (char *)&_ebss);
}

uint32_t usedStack() {
    // Stack grows DOWN from _estack.
    // Used stack = _estack - SP
    auto sp = (char *)__builtin_frame_address(0);
    return (uint32_t)((char *)&_estack - sp);
}

uint32_t freeHeap() {
    // Heap lives in RAM2 (OCRAM2), starting after DMAMEM variables.
    // Free heap = _heap_end - __brkval
    //
    // __brkval is a pointer variable stored in DTCM, but its VALUE
    // points into RAM2 (>= _heap_start = 0x2025A8A0).
    return (uint32_t)((char *)&_heap_end - __brkval);
}

uint32_t freeRAM2() {
    // RAM2 = DMAMEM statics (fixed) + heap (dynamic).
    // The only allocatable free space is the heap portion.
    return freeHeap();
}


// ─── Stack Watermark ─────────────────────────────────────────────────
//
// How it works:
//   1. paintStackRegion() writes a known pattern below current SP
//   2. The computation's stack frames overwrite the pattern from the
//      TOP (high address) downward as the stack grows
//   3. checkStackWaterMark() scans from BOTTOM (low address) up to
//      find where the pattern was destroyed → peak stack depth
//
// For accurate results, call paint/check at the SAME call depth as
// the computation (not from a sub-function), to avoid measurement gaps.

uint32_t* paintStackRegion(uint32_t paint_bytes) {
    // Use a volatile local to approximate current SP
    volatile uint32_t anchor;
    uint32_t sp_approx = (uint32_t)&anchor;

    // Paint BELOW current SP (lower addresses = where future stack goes)
    // Align base to 4-byte boundary
    uint32_t *base = (uint32_t *)((sp_approx - paint_bytes) & ~0x3u);

    uint32_t count = paint_bytes / sizeof(uint32_t);
    for (uint32_t i = 0; i < count; i++) {
        base[i] = STACK_PAINT_PATTERN;
    }

    return base;
}

uint32_t checkStackWaterMark(const uint32_t *painted_base, uint32_t paint_bytes) {
    uint32_t count = paint_bytes / sizeof(uint32_t);

    // Scan from BOTTOM (lowest address, idx 0) upward.
    // Untouched words still have our pattern; the first non-pattern word
    // means the stack reached that point.
    uint32_t untouched_words = 0;
    for (uint32_t i = 0; i < count; i++) {
        if (painted_base[i] == STACK_PAINT_PATTERN) {
            untouched_words++;
        } else {
            break;
        }
    }

    // Peak stack into painted area = total painted - untouched portion
    return paint_bytes - (untouched_words * sizeof(uint32_t));
}


// ─── MemoryTracker Implementation ───────────────────────────────────

void MemoryTracker::Reset(uint32_t input_buf_alloc, uint32_t output_buf_alloc) {
    count_ = 0;
    input_buf_alloc_  = input_buf_alloc;
    output_buf_alloc_ = output_buf_alloc;

    baseline_free_ram1_ = freeRAM1();
    baseline_free_ram2_ = freeRAM2();

    memset(snapshots_, 0, sizeof(snapshots_));
}


void MemoryTracker::RecordBeforeLayer(uint32_t layer_in_block, uint32_t layer_idx,
                                       uint32_t input_data_bytes,
                                       uint32_t output_data_bytes,
                                       uint32_t weight_bytes) {
    if (layer_in_block >= MAX_BLOCK_LAYERS) return;

    LayerMemSnapshot &snap = snapshots_[layer_in_block];
    snap.layer_idx         = layer_idx;
    snap.input_data_bytes  = input_data_bytes;
    snap.output_data_bytes = output_data_bytes;
    snap.weight_bytes      = weight_bytes;

    // Ping-pong mapping:
    //   Even layer_in_block → src = input_buffer_ (RAM1), dst = output_buffer_ (RAM2)
    //   Odd  layer_in_block → src = output_buffer_ (RAM2), dst = input_buffer_ (RAM1)
    if (layer_in_block % 2 == 0) {
        snap.ram1_data_bytes = input_data_bytes;   // src in RAM1
        snap.ram2_data_bytes = output_data_bytes;  // dst in RAM2
    } else {
        snap.ram1_data_bytes = output_data_bytes;  // dst in RAM1
        snap.ram2_data_bytes = input_data_bytes;   // src in RAM2
    }

    snap.free_ram1_before  = freeRAM1();
    snap.stack_peak_bytes  = 0;

    if (layer_in_block >= count_) {
        count_ = layer_in_block + 1;
    }
}

void MemoryTracker::RecordAfterLayer(uint32_t layer_in_block, uint32_t stack_peak_bytes) {
    if (layer_in_block >= MAX_BLOCK_LAYERS) return;

    LayerMemSnapshot &snap = snapshots_[layer_in_block];
    snap.free_ram1_after  = freeRAM1();
    snap.stack_peak_bytes = stack_peak_bytes;
}


// ─── Analysis ────────────────────────────────────────────────────────

uint32_t MemoryTracker::GetEffectivePeakRAM1() const {
    // Effective RAM1 peak = max over layers of (data_in_RAM1 + stack_during_compute)
    // Both consume RAM1 (DTCM): data is in input_buffer_ region, stack at top.
    uint32_t peak = 0;
    for (uint8_t i = 0; i < count_; i++) {
        uint32_t total = snapshots_[i].ram1_data_bytes + snapshots_[i].stack_peak_bytes;
        if (total > peak) {
            peak = total;
        }
    }
    return peak;
}

uint32_t MemoryTracker::GetEffectivePeakRAM2() const {
    // RAM2 has no stack — only tensor data in output_buffer_ region.
    uint32_t peak = 0;
    for (uint8_t i = 0; i < count_; i++) {
        if (snapshots_[i].ram2_data_bytes > peak) {
            peak = snapshots_[i].ram2_data_bytes;
        }
    }
    return peak;
}

uint32_t MemoryTracker::GetEffectivePeakTotal() const {
    // Per-layer: ram1_data + stack + ram2_data.  Take the max.
    uint32_t peak = 0;
    for (uint8_t i = 0; i < count_; i++) {
        uint32_t combined = snapshots_[i].ram1_data_bytes
                          + snapshots_[i].stack_peak_bytes
                          + snapshots_[i].ram2_data_bytes;
        if (combined > peak) {
            peak = combined;
        }
    }
    return peak;
}

uint32_t MemoryTracker::GetWasteRAM1() const {
    uint32_t eff = GetEffectivePeakRAM1();
    return (input_buf_alloc_ > eff) ? (input_buf_alloc_ - eff) : 0;
}

uint32_t MemoryTracker::GetWasteRAM2() const {
    uint32_t eff = GetEffectivePeakRAM2();
    return (output_buf_alloc_ > eff) ? (output_buf_alloc_ - eff) : 0;
}

uint32_t MemoryTracker::GetWasteTotal() const {
    return GetWasteRAM1() + GetWasteRAM2();
}

float MemoryTracker::GetUtilizationPercent() const {
    uint32_t phys = GetPhysicalAllocTotal();
    if (phys == 0) return 0.0f;
    return (float)GetEffectivePeakTotal() / (float)phys * 100.0f;
}

float MemoryTracker::GetLayerUtilization(uint8_t idx) const {
    if (idx >= count_) return 0.0f;
    uint32_t phys = input_buf_alloc_ + output_buf_alloc_;
    if (phys == 0) return 0.0f;
    uint32_t used = snapshots_[idx].ram1_data_bytes
                  + snapshots_[idx].stack_peak_bytes
                  + snapshots_[idx].ram2_data_bytes;
    return (float)used / (float)phys * 100.0f;
}

uint32_t MemoryTracker::GetPeakLayerIdx() const {
    uint32_t peak = 0;
    uint32_t peak_layer = 0;
    for (uint8_t i = 0; i < count_; i++) {
        uint32_t combined = snapshots_[i].ram1_data_bytes
                          + snapshots_[i].stack_peak_bytes
                          + snapshots_[i].ram2_data_bytes;
        if (combined > peak) {
            peak = combined;
            peak_layer = snapshots_[i].layer_idx;
        }
    }
    return peak_layer;
}

uint32_t MemoryTracker::GetMaxStackPeak() const {
    uint32_t peak = 0;
    for (uint8_t i = 0; i < count_; i++) {
        if (snapshots_[i].stack_peak_bytes > peak) {
            peak = snapshots_[i].stack_peak_bytes;
        }
    }
    return peak;
}


// ─── Output ──────────────────────────────────────────────────────────

void MemoryTracker::PrintReport() const {
    Serial.println("╔══════════════════════════════════════════════════════════╗");
    Serial.println("║            Memory Tracker — Block Summary               ║");
    Serial.println("╠══════════════════════════════════════════════════════════╣");

    Serial.printf( "║  Layers tracked:      %u\n", count_);
    Serial.printf( "║  Baseline free:       RAM1(DTCM)=%u B (%u KB)\n",
                   baseline_free_ram1_, baseline_free_ram1_ / 1024);
    Serial.printf( "║                       RAM2(heap)=%u B (%u KB)\n",
                   baseline_free_ram2_, baseline_free_ram2_ / 1024);
    Serial.printf( "║  Stack used at start: %u B\n", usedStack());

    Serial.println("╠──────────────────────────────────────────────────────────╣");

    Serial.println("║  Physical (allocated buffer sizes):");
    Serial.printf( "║    RAM1 input_buffer_  = %u KB\n", input_buf_alloc_ / 1024);
    Serial.printf( "║    RAM2 output_buffer_ = %u KB\n", output_buf_alloc_ / 1024);
    Serial.printf( "║    Total buffers       = %u KB\n", GetPhysicalAllocTotal() / 1024);

    Serial.println("║  Effective peak (min buffer + stack needed):");
    Serial.printf( "║    RAM1 peak (data+stack) = %u B (%u KB)\n",
                   GetEffectivePeakRAM1(), GetEffectivePeakRAM1() / 1024);
    Serial.printf( "║    RAM2 peak (data only)  = %u B (%u KB)\n",
                   GetEffectivePeakRAM2(), GetEffectivePeakRAM2() / 1024);
    Serial.printf( "║    Combined peak          = %u B (%u KB)\n",
                   GetEffectivePeakTotal(), GetEffectivePeakTotal() / 1024);

    Serial.println("║  Stack:");
    Serial.printf( "║    Max stack peak (compute) = %u B\n", GetMaxStackPeak());
    Serial.printf( "║    Free RAM1 (for stack)    = %u B (%u KB)\n",
                   freeRAM1(), freeRAM1() / 1024);

    Serial.println("║  Waste (alloc buffer − effective data):");
    Serial.printf( "║    RAM1 waste = %u KB\n", GetWasteRAM1() / 1024);
    Serial.printf( "║    RAM2 waste = %u KB\n", GetWasteRAM2() / 1024);
    Serial.printf( "║  Utilization  = %.1f%%\n", GetUtilizationPercent());
    Serial.printf( "║  Peak layer   = model_layer_config[%u]\n", GetPeakLayerIdx());

    Serial.println("╚══════════════════════════════════════════════════════════╝");
}

void MemoryTracker::PrintLayerDetail() const {
    Serial.println("┌─────┬───────┬──────────┬──────────┬──────────┬──────────┬──────────┬───────┬───────┐");
    Serial.println("│ #   │ Layer │ In Bytes │ Out Bytes│ RAM1 Use │ RAM2 Use │ Stk Peak │ Wt(KB)│ Util% │");
    Serial.println("├─────┼───────┼──────────┼──────────┼──────────┼──────────┼──────────┼───────┼───────┤");

    for (uint8_t i = 0; i < count_; i++) {
        const auto &s = snapshots_[i];
        Serial.printf("│ %3u │ %5u │ %8u │ %8u │ %8u │ %8u │ %8u │ %5u │ %5.1f │\n",
                       i, s.layer_idx,
                       s.input_data_bytes, s.output_data_bytes,
                       s.ram1_data_bytes, s.ram2_data_bytes,
                       s.stack_peak_bytes,
                       s.weight_bytes / 1024,
                       GetLayerUtilization(i));
    }

    Serial.println("└─────┴───────┴──────────┴──────────┴──────────┴──────────┴──────────┴───────┴───────┘");

    // Free RAM1 deltas (shows if any heap/stack leak between layers)
    Serial.println("  Free RAM1 per layer (before -> after):");
    for (uint8_t i = 0; i < count_; i++) {
        const auto &s = snapshots_[i];
        int32_t delta = (int32_t)s.free_ram1_after - (int32_t)s.free_ram1_before;
        Serial.printf("    Layer %u: %u -> %u  (delta = %+d B, stack peak = %u B)\n",
                       s.layer_idx, s.free_ram1_before, s.free_ram1_after,
                       delta, s.stack_peak_bytes);
    }
}

} // namespace mem