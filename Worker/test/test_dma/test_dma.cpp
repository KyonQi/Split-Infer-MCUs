#include <Arduino.h>
#include <DMAChannel.h>

// Simulate 1MB of data to process
#define TOTAL_DATA_SIZE (1024 * 1024) 

// Each chunk is 16KB 
#define CHUNK_SIZE      (16 * 1024)   

// Number of iterations in the compute function to simulate processing time
#define COMPUTE_ITERATIONS 20 

// ================= Memory Allocation =================
// Souce data placed in Flash (2x faster with DMA)
// there's no big diff if you place it in DMARAM(i.e., RAM2)
const uint8_t sourceData[TOTAL_DATA_SIZE] PROGMEM __attribute__((aligned(32))) = {0};

// Double buffers for DMA, placed in RAM(1)
uint8_t bufferA[CHUNK_SIZE] __attribute__((aligned(32)));
uint8_t bufferB[CHUNK_SIZE] __attribute__((aligned(32)));

DMAChannel dma;

// ================= Helper Functions =================
// Check if DMA channel is active by reading the TCD Control and Status Register (CSR)
// Reference: i.MX RT Reference Manual, TCD Control and Status (CSR) Register
// Bit 6 (ACTIVE): Channel Active. 1 = channel is executing.
bool is_dma_active() {
    return (dma.TCD->CSR & 0x0020) ? true : false;
}

// Simulate compute task
FASTRUN uint32_t process_chunk(const uint8_t* data, size_t size) {
    uint32_t sum = 0;
    for (int k = 0; k < COMPUTE_ITERATIONS; k++) {
        for (size_t i = 0; i < size; i += 4) {
            sum += *(uint32_t*)(data + i);
            sum ^= k;
        }
    }
    return sum;
}

void setup_data() {
    // for (int i = 0; i < TOTAL_DATA_SIZE; i++) {
    //     sourceData[i] = (uint8_t)(i & 0xFF);
    // }
    // arm_dcache_flush(sourceData, TOTAL_DATA_SIZE);
}

// ================= Test 1: Sequential Processing (Baseline) =================
void test_sequential() {
    Serial.println("\n--- Starting Sequential Test (CPU Copy + Compute) ---");
    uint32_t total_checksum = 0;
    
    unsigned long start_time = micros();

    for (int offset = 0; offset < TOTAL_DATA_SIZE; offset += CHUNK_SIZE) {
        // 1. CPU Copy 
        memcpy(bufferA, sourceData + offset, CHUNK_SIZE);
        
        // 2. Compute
        total_checksum += process_chunk(bufferA, CHUNK_SIZE);
    }

    unsigned long end_time = micros();
    Serial.printf("Time: %lu us | Checksum: %u\n", end_time - start_time, total_checksum);
}

// ================= Test 2: DMA Double Buffering (Fixed) =================
void test_double_buffered_dma() {
    Serial.println("\n--- Starting DMA Double Buffering Test ---");
    uint32_t total_checksum = 0;
    
    // Configure DMA
    // disableOnCompletion(): Every time the DMA finishes a block (BITER reaches 0), it will automatically clear the ERQ (Enable Request) bit, preventing further triggers.
    // But we are using triggerManual(), which ignores ERQ and starts directly.
    // Here we are more concerned that the ACTIVE bit will be automatically cleared after the transfer is complete.
    dma.disableOnCompletion(); 
    
    unsigned long start_time = micros();

    int total_chunks = TOTAL_DATA_SIZE / CHUNK_SIZE;
    
    uint8_t* compute_buf = nullptr;
    uint8_t* fill_buf = nullptr;

    // --- Start (Pipeline Filling) ---
    // 1. Start DMA transfer of chunk 0 to Buffer A
    dma.sourceBuffer(sourceData, CHUNK_SIZE);
    dma.destinationBuffer(bufferA, CHUNK_SIZE);
    
    // [Fix] Use triggerManual
    dma.triggerManual(); 
    
    // [Fix] Manually check the ACTIVE bit in TCD->CSR
    while (is_dma_active()); 
    
    // After DMA transfer completes, clear Cache
    arm_dcache_delete(bufferA, CHUNK_SIZE);

    // --- Steady State ---
    for (int i = 1; i < total_chunks; i++) {
        // Ping-pong logic
        if (i % 2 != 0) {
            compute_buf = bufferA;
            fill_buf    = bufferB;
        } else {
            compute_buf = bufferB;
            fill_buf    = bufferA;
        }

        // [DMA Action]: Configure and start transfer of next chunk (i) -> fill_buf
        dma.sourceBuffer(sourceData + (i * CHUNK_SIZE), CHUNK_SIZE);
        dma.destinationBuffer(fill_buf, CHUNK_SIZE);
        
        // [Fix] Start transfer
        dma.triggerManual(); 

        // [CPU Action]: Process the current chunk (i-1) -> compute_buf
        total_checksum += process_chunk(compute_buf, CHUNK_SIZE);

        // [Sync Point]: Wait for DMA to complete
        // If the CPU is faster, we must wait here for DMA
        while (is_dma_active()) {
            asm("nop"); // Prevent compiler from over-optimizing empty loop
        }

        // Key: Invalidate Cache
        arm_dcache_delete(fill_buf, CHUNK_SIZE);
    }

    // --- Ending Phase (Pipeline Draining) ---
    // Process the last chunk
    if (total_chunks % 2 != 0) {
        compute_buf = bufferA; 
    } else {
        compute_buf = bufferB;
    }
    total_checksum += process_chunk(compute_buf, CHUNK_SIZE);

    unsigned long end_time = micros();
    Serial.printf("Time: %lu us | Checksum: %u\n", end_time - start_time, total_checksum);
}

void setup() {
    Serial.begin(115200);
    while (!Serial && millis() < 4000); 
    
    Serial.printf("Teensy 4.1 DMA Double Buffer Test\n");
    Serial.printf("Core Clock: %u MHz\n", F_CPU_ACTUAL / 1000000);
    Serial.printf("Data Size: %d KB\n", TOTAL_DATA_SIZE / 1024);
    Serial.printf("Chunk Size: %d KB\n", CHUNK_SIZE / 1024);

    setup_data();

    delay(1000);
    test_sequential();
    delay(1000);
    test_double_buffered_dma();
    
    Serial.println("\nDone.");
}

void loop() {
    digitalWrite(LED_BUILTIN, !digitalRead(LED_BUILTIN));
    delay(500);
}