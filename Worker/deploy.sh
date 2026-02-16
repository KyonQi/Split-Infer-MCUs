#!/usr/bin/env bash
# deploy.sh - Flash multiple Teensy 4.1 workers one-at-a-time
#
# Designed for single-USB-port setups: builds ALL firmwares first,
# then guides you through plugging in each Teensy one by one.
# Notice that you need to specify the same num_wokers as you did in Python_Sim_Infer exporter.
#
# Usage:
#   ./deploy.sh                    # deploy all workers (0..NUM_WORKERS-1)
#   ./deploy.sh 0 2               # deploy only worker 0 and 2
#   ./deploy.sh --build-only       # only build, skip upload
#
# Flow:
#   Phase 1: Build firmware for all workers (no hardware needed)
#   Phase 2: For each worker, prompt you to plug in -> detect -> upload -> unplug

set -euo pipefail

# ═══════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════
COORD_IP_0=192
COORD_IP_1=168
COORD_IP_2=1
COORD_IP_3=10
COORD_PORT=54321
NUM_WORKERS=4
PIO_ENV="deploy"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Where PlatformIO puts build artifacts
BUILD_DIR=".pio/build/${PIO_ENV}"

# Per-worker firmware cache directory
FW_CACHE_DIR=".pio/firmware_cache"
mkdir -p "$FW_CACHE_DIR"

# ═══════════════════════════════════════════════════════════════
# Parse arguments
# ═══════════════════════════════════════════════════════════════
BUILD_ONLY=false
DEPLOY_IDS=()

for arg in "$@"; do
    if [[ "$arg" == "--build-only" ]]; then
        BUILD_ONLY=true
    else
        DEPLOY_IDS+=("$arg")
    fi
done

if [[ ${#DEPLOY_IDS[@]} -eq 0 ]]; then
    DEPLOY_IDS=($(seq 0 $((NUM_WORKERS - 1))))
fi

echo "╔════════════════════════════════════════════════╗"
echo "║       Multi-MCU Worker Deployment Tool         ║"
echo "╠════════════════════════════════════════════════╣"
echo "║  Coordinator: ${COORD_IP_0}.${COORD_IP_1}.${COORD_IP_2}.${COORD_IP_3}:${COORD_PORT}║"
echo "║  Workers:     ${DEPLOY_IDS[*]}║"
echo "║  Mode:        $(if $BUILD_ONLY; then echo 'Build only'; else echo 'Build + Upload'; fi)║"
echo "╚════════════════════════════════════════════════╝"
echo ""

# ═══════════════════════════════════════════════════════════════
# Pre-flight: verify per-worker headers exist
# ═══════════════════════════════════════════════════════════════
HEADER_BASE="${SCRIPT_DIR}/../../Python_Sim_Infer/extractor_mcu"
for wid in "${DEPLOY_IDS[@]}"; do
    MCU_DIR="${HEADER_BASE}/mcu_${wid}"
    if [[ ! -d "$MCU_DIR" ]]; then
        echo "ERROR: Header directory not found: ${MCU_DIR}"
        echo "  Run the Python exporter first:"
        echo "    cd Python_Sim_Infer && python -m extractor_mcu.export_weights --num_mcus ${NUM_WORKERS}"
        exit 1
    fi
    for hf in weights.h quant_params.h layer_config.h; do
        if [[ ! -f "${MCU_DIR}/${hf}" ]]; then
            echo "ERROR: Missing header: ${MCU_DIR}/${hf}"
            exit 1
        fi
    done
done
echo "[✓] All per-worker header files found."
echo ""

# ═══════════════════════════════════════════════════════════════
# Phase 1: Build firmware for ALL workers (no hardware needed)
# ═══════════════════════════════════════════════════════════════
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Phase 1: Building firmware for all workers"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

for wid in "${DEPLOY_IDS[@]}"; do
    echo ""
    echo "[Worker ${wid}] Building..."

    export EXTRA_BUILD_FLAGS="-DWORKER_ID=${wid} -DSVR_IP_0=${COORD_IP_0} -DSVR_IP_1=${COORD_IP_1} -DSVR_IP_2=${COORD_IP_2} -DSVR_IP_3=${COORD_IP_3} -DSVR_PORT=${COORD_PORT}"

    # Full clean to avoid stale objects (each worker has different weights.h)
    pio run -e "$PIO_ENV" -t clean > /dev/null 2>&1

    # Build
    if ! pio run -e "$PIO_ENV" 2>&1 | tail -3; then
        echo "[Worker ${wid}] BUILD FAILED"
        exit 1
    fi

    # Cache the firmware binary for later upload
    # Teensy uses .hex format
    FW_FILE="${BUILD_DIR}/firmware.hex"
    if [[ ! -f "$FW_FILE" ]]; then
        # Some PlatformIO versions use .elf or other names
        FW_FILE=$(find "$BUILD_DIR" -maxdepth 1 -name "*.hex" -o -name "*.bin" | head -1)
    fi

    if [[ -z "$FW_FILE" || ! -f "$FW_FILE" ]]; then
        echo "[Worker ${wid}] ERROR: Cannot find built firmware in ${BUILD_DIR}"
        ls -la "$BUILD_DIR"/
        exit 1
    fi

    cp "$FW_FILE" "${FW_CACHE_DIR}/worker_${wid}.hex"
    echo "[Worker ${wid}] Build OK → cached as worker_${wid}.hex"
done

echo ""
echo "[✓] All firmware images built successfully."
echo ""

if $BUILD_ONLY; then
    echo "Build-only mode: skipping upload phase."
    echo "Cached firmware files:"
    ls -lh "${FW_CACHE_DIR}/"
    exit 0
fi

# ═══════════════════════════════════════════════════════════════
# Phase 2: Upload firmware one-by-one (plug, flash, unplug)
# ═══════════════════════════════════════════════════════════════
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Phase 2: Upload firmware"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "  Instructions:"
echo "    1. Plug in the Teensy for the worker when prompted"
echo "    2. Wait for upload to complete"
echo "    3. Unplug and plug in the next one"
echo ""

# Function: wait for a Teensy to appear as a USB device
# Sets global DETECTED_PORT variable
wait_for_teensy() {
    DETECTED_PORT=""
    echo "    Waiting for Teensy to appear..."
    local timeout=60
    local elapsed=0
    while [[ $elapsed -lt $timeout ]]; do
        local port
        port=$(find /dev -maxdepth 1 -name "ttyACM*" 2>/dev/null | head -1)
        if [[ -n "$port" ]]; then
            echo "    Detected: $port"
            DETECTED_PORT="$port"
            return 0
        fi
        sleep 1
        elapsed=$((elapsed + 1))
        if (( elapsed % 5 == 0 )); then
            printf "    Still waiting... (%ds)\n" "$elapsed"
        fi
    done
    echo "    Timeout: No Teensy detected after ${timeout}s"
    return 1
}

# Function: wait for Teensy to be removed
wait_for_removal() {
    echo "    Waiting for Teensy to be unplugged..."
    while true; do
        local port
        port=$(find /dev -maxdepth 1 -name "ttyACM*" 2>/dev/null | head -1)
        if [[ -z "$port" ]]; then
            return 0
        fi
        sleep 1
    done
}

SUCCESS=()
FAILED=()

for wid in "${DEPLOY_IDS[@]}"; do
    echo "────────────────────────────────────────────────"
    echo "  Worker ${wid}: Plug in the Teensy now"
    echo "────────────────────────────────────────────────"

    # Check if Teensy is already connected
    EXISTING_PORT=$(find /dev -maxdepth 1 -name "ttyACM*" 2>/dev/null | head -1)
    if [[ -n "$EXISTING_PORT" ]]; then
        PORT="$EXISTING_PORT"
        echo "    Teensy already connected at: $PORT"
    else
        if wait_for_teensy; then
            PORT="$DETECTED_PORT"
        else
            echo "[Worker ${wid}] SKIPPED: No device detected"
            FAILED+=("$wid")
            continue
        fi
    fi

    # Brief settle time after USB enumeration
    sleep 1

    FW_HEX="${FW_CACHE_DIR}/worker_${wid}.hex"
    echo "[Worker ${wid}] Uploading ${FW_HEX} → ${PORT}..."

    # Copy cached hex back to build dir so PlatformIO uploads the correct firmware
    cp "$FW_HEX" "${BUILD_DIR}/firmware.hex"

    # Teensy upload uses teensy_loader_cli via USB HID, not serial port.
    # When only one Teensy is connected, PlatformIO auto-detects it.
    # You may need to press the button on the Teensy to enter bootloader mode.
    export EXTRA_BUILD_FLAGS="-DWORKER_ID=${wid}"  # needed for pio to resolve env
    if pio run -e "$PIO_ENV" -t upload 2>&1 | tail -5; then
        echo "[Worker ${wid}] ✓ Upload SUCCESS"
        SUCCESS+=("$wid")
    else
        echo "[Worker ${wid}] ✗ Upload FAILED"
        FAILED+=("$wid")
    fi

    # If there are more workers to flash, prompt to unplug
    # Find remaining workers
    REMAINING=()
    _found=false
    for rid in "${DEPLOY_IDS[@]}"; do
        if $_found; then
            REMAINING+=("$rid")
        fi
        if [[ "$rid" == "$wid" ]]; then
            _found=true
        fi
    done

    if [[ ${#REMAINING[@]} -gt 0 ]]; then
        echo ""
        echo "    Done with Worker ${wid}. Please UNPLUG this Teensy."
        echo "    Remaining: ${REMAINING[*]}"
        wait_for_removal
        echo "    Teensy removed. Ready for next one."
        echo ""
        sleep 1
    fi
done

# ═══════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════
echo ""
echo "╔════════════════════════════════════════════════╗"
echo "║            Deployment Summary                  ║"
echo "╠════════════════════════════════════════════════╣"
echo "║  Success: id_${SUCCESS[*]:-none}"
echo "║  Failed:  ${FAILED[*]:-none}"
echo "╚════════════════════════════════════════════════╝"

if [[ ${#FAILED[@]} -gt 0 ]]; then
    exit 1
fi