#ifndef MEMORY_TRACKER_H
#define MEMORY_TRACKER_H

#include <Arduino.h>
#include <stdint.h>

// ═══════════════════════════════════════════════════════════════════════
//  Teensy 4.1 Memory Map  (verified from ELF symbols)
// ═══════════════════════════════════════════════════════════════════════
//
//  RAM1 (DTCM) 416KB usable @ 0x20000000:
//    [.data] [.bss (incl. input_buffer_ 350KB)] [_ebss] ... FREE ~21KB ... [_estack]
//    ← heap is NOT here; only static data + stack →
//
//  RAM2 (OCRAM2) 512KB @ 0x20200000:
//    [DMAMEM (incl. output_buffer_ 350KB)] [_heap_start] ... HEAP ... [_heap_end]
//    ← heap lives here (malloc/new allocate from RAM2) →
//
//  Flash 8MB @ 0x60000000:
//    Code, PROGMEM const data (weights, quant params)
//
// ═══════════════════════════════════════════════════════════════════════

namespace mem {

// ─── Constants ───────────────────────────────────────────────────────

static constexpr uint32_t STACK_PAINT_PATTERN = 0xDEADC0DE;
static constexpr uint32_t DEFAULT_PAINT_BYTES = 4096; // 4KB default

// ─── Low-Level RAM Query ─────────────────────────────────────────────

/// Free RAM1 (DTCM): gap between end-of-BSS (_ebss) and current SP.
/// On Teensy 4.1, RAM1 has NO heap — only static data (.data, .bss) and stack.
/// Free RAM1 = SP - _ebss
///   This is ALL the available space for stack growth.
uint32_t freeRAM1();

/// Used stack: how far SP has grown down from _estack (top of DTCM).
/// used_stack = _estack - SP
uint32_t usedStack();

/// Free heap (in RAM2/OCRAM2): how much more malloc/new can allocate.
/// free_heap = _heap_end - __brkval
/// On Teensy 4.1, the heap resides in RAM2 (after DMAMEM static vars).
uint32_t freeHeap();

/// Free RAM2 total ≈ freeHeap().
/// RAM2 layout: [DMAMEM statics] [heap→__brkval] [free] [_heap_end]
/// Since DMAMEM is fixed at compile time, free RAM2 = free heap.
uint32_t freeRAM2();

// ─── Stack Watermark ─────────────────────────────────────────────────
//
// IMPORTANT: For accurate measurement, call paintStackRegion() from the
// SAME call depth as the computation you want to measure (or one level
// above). The painted region starts just below the current SP, and the
// computation's stack frames will overwrite the painted area from the top
// (high address) downward.
//
// Usage pattern (in HandleBlockComputing):
//
//   uint32_t* paint_base = mem::paintStackRegion(4096);
//   // ... conv2d computation ...
//   uint32_t peak = mem::checkStackWaterMark(paint_base, 4096);
//

/// Paint `paint_bytes` below current SP with a known bit-pattern.
/// Returns pointer to the base (lowest address) of the painted region.
uint32_t* paintStackRegion(uint32_t paint_bytes = DEFAULT_PAINT_BYTES);

/// After computation, scan the painted region to see how deep the
/// stack descended. Returns peak stack usage in bytes.
uint32_t checkStackWaterMark(const uint32_t* painted_base,
                              uint32_t paint_bytes = DEFAULT_PAINT_BYTES);


// ═══════════════════════════════════════════════════════════════════════
//  Per-Layer Memory Snapshot
// ═══════════════════════════════════════════════════════════════════════

struct LayerMemSnapshot {
    uint32_t layer_idx;          // absolute index in model_layer_config[]

    // ── Tensor Data Sizes (actual payload, not 350KB buffer size) ──
    uint32_t input_data_bytes;   // = in_C × in_H × in_W  (uint8 quantized)
    uint32_t output_data_bytes;  // = out_C × out_H × out_W
    uint32_t weight_bytes;       // model_weights[idx].weights_size (in Flash)

    // ── Per-RAM Actual Data (ping-pong aware) ──
    // Even layer_in_block: src = input_buffer_ (RAM1), dst = output_buffer_ (RAM2)
    // Odd  layer_in_block: src = output_buffer_ (RAM2), dst = input_buffer_ (RAM1)
    uint32_t ram1_data_bytes;    // tensor data residing in RAM1 this layer
    uint32_t ram2_data_bytes;    // tensor data residing in RAM2 this layer

    // ── Runtime Measurements ──
    uint32_t free_ram1_before;   // freeRAM1() just before compute
    uint32_t free_ram1_after;    // freeRAM1() just after compute
    uint32_t stack_peak_bytes;   // peak stack usage during compute (via watermark)
};


// ═══════════════════════════════════════════════════════════════════════
//  Memory Tracker
// ═══════════════════════════════════════════════════════════════════════
//
//  Integration in HandleBlockComputing():
//
//    mem_tracker_.Reset(sizeof(input_buffer_), sizeof(output_buffer_));
//
//    for (uint32_t idx = start_idx; idx <= end_idx; idx++) {
//        uint32_t lib = idx - start_idx; // layer_in_block
//        mem_tracker_.RecordBeforeLayer(lib, idx, in_bytes, out_bytes, w_bytes);
//
//        // Paint stack HERE — same call depth as conv2d for accuracy
//        uint32_t* paint = mem::paintStackRegion();
//        conv2d(...);
//        uint32_t stk = mem::checkStackWaterMark(paint);
//
//        mem_tracker_.RecordAfterLayer(lib, stk);
//    }
//    mem_tracker_.PrintReport();
//
// ═══════════════════════════════════════════════════════════════════════

static constexpr uint8_t MAX_BLOCK_LAYERS = 16;

class MemoryTracker final {
public:
    MemoryTracker() = default;

    // ─── Lifecycle ───

    /// Call at the start of each block/task.
    /// input_buf_alloc / output_buf_alloc: the ALLOCATED sizes of the
    ///   global buffers (e.g. 350*1024 each).
    void Reset(uint32_t input_buf_alloc, uint32_t output_buf_alloc);

    // ─── Per-Layer Recording ───

    /// Call immediately BEFORE a layer's compute function.
    void RecordBeforeLayer(uint32_t layer_in_block, uint32_t layer_idx,
                           uint32_t input_data_bytes,
                           uint32_t output_data_bytes,
                           uint32_t weight_bytes);

    /// Call immediately AFTER a layer's compute function.
    /// stack_peak_bytes: result from checkStackWaterMark() done by the caller,
    ///   or 0 if stack measurement was not performed for this layer.
    void RecordAfterLayer(uint32_t layer_in_block, uint32_t stack_peak_bytes = 0);

    // ─── Analysis: Effective vs Physical ───

    /// Effective peak RAM1 = max over layers of (ram1_data_bytes + stack_peak_bytes)
    /// This is the minimum RAM1 needed if buffers were precisely sized,
    /// accounting for both tensor data and compute stack overhead.
    uint32_t GetEffectivePeakRAM1() const;

    /// Effective peak RAM2 = max over layers of (ram2_data_bytes)
    /// No stack in RAM2; only tensor data.
    uint32_t GetEffectivePeakRAM2() const;

    /// Effective peak total = max per-layer (ram1_data + stack + ram2_data)
    uint32_t GetEffectivePeakTotal() const;

    /// Physical: the actually allocated buffer sizes (constant 350KB each).
    uint32_t GetPhysicalAllocRAM1() const { return input_buf_alloc_; }
    uint32_t GetPhysicalAllocRAM2() const { return output_buf_alloc_; }
    uint32_t GetPhysicalAllocTotal() const { return input_buf_alloc_ + output_buf_alloc_; }

    /// Waste = physical buffer alloc - effective data peak
    uint32_t GetWasteRAM1() const;
    uint32_t GetWasteRAM2() const;
    uint32_t GetWasteTotal() const;

    /// Buffer utilization = effective / physical * 100
    float GetUtilizationPercent() const;

    // ─── Per-Layer Queries ───

    float GetLayerUtilization(uint8_t snapshot_idx) const;
    uint32_t GetPeakLayerIdx() const;
    uint32_t GetMaxStackPeak() const;

    // ─── Output ───

    void PrintReport() const;
    void PrintLayerDetail() const;

    // ─── Accessors ───

    uint8_t GetCount() const { return count_; }
    const LayerMemSnapshot& GetSnapshot(uint8_t idx) const { return snapshots_[idx]; }
    uint32_t GetBaselineFreeRAM1() const { return baseline_free_ram1_; }
    uint32_t GetBaselineFreeRAM2() const { return baseline_free_ram2_; }

private:
    LayerMemSnapshot snapshots_[MAX_BLOCK_LAYERS]{};
    uint8_t count_ = 0;

    uint32_t input_buf_alloc_  = 0;   // RAM1 buffer allocation (input_buffer_)
    uint32_t output_buf_alloc_ = 0;   // RAM2 buffer allocation (output_buffer_)

    uint32_t baseline_free_ram1_ = 0; // freeRAM1() at Reset() time
    uint32_t baseline_free_ram2_ = 0; // freeRAM2() at Reset() time
};

} // namespace mem

#endif // MEMORY_TRACKER_H