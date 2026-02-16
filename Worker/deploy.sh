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
#   ./deploy.sh --clean            # clean build artifacts and firmware cache
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
DO_CLEAN=false
DEPLOY_IDS=()

for arg in "$@"; do
    case "$arg" in
        --build-only)  BUILD_ONLY=true ;;
        --clean)       DO_CLEAN=true ;;
        *)             DEPLOY_IDS+=("$arg") ;;
    esac
done

# ═══════════════════════════════════════════════════════════════
# Handle clean
# ═══════════════════════════════════════════════════════════════
if $DO_CLEAN; then
    echo "[clean] Cleaning build artifacts for env:${PIO_ENV}..."
    export EXTRA_BUILD_FLAGS="-DWORKER_ID=0"  # dummy, needed for pio to parse env
    pio run -e "$PIO_ENV" -t clean 2>&1 | tail -3
    echo "[clean] Removing firmware cache..."
    rm -rf "${FW_CACHE_DIR}"
    echo "[clean] Done."
    if ! $BUILD_ONLY && [[ ${#DEPLOY_IDS[@]} -eq 0 ]]; then
        exit 0
    fi
fi

if [[ ${#DEPLOY_IDS[@]} -eq 0 ]]; then
    DEPLOY_IDS=($(seq 0 $((NUM_WORKERS - 1))))
fi

# Ensure cache dir exists (may have been removed by --clean)
mkdir -p "$FW_CACHE_DIR"

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
echo "  Phase 2: Upload firmware (interactive)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "  You will be prompted before each upload."
echo "  Plug in the correct Teensy and press ENTER when ready."
echo ""

SUCCESS=()
FAILED=()

for wid in "${DEPLOY_IDS[@]}"; do
    echo "────────────────────────────────────────────────"
    echo "  Worker ${wid}: Ready to flash"
    echo "────────────────────────────────────────────────"
    echo ""
    read -rp "    Plug in the Teensy for Worker ${wid}, then press ENTER to continue... "
    echo ""

    # Brief settle time after USB enumeration
    sleep 1

    # Detect port
    PORT=$(find /dev -maxdepth 1 -name "ttyACM*" 2>/dev/null | head -1)
    if [[ -z "$PORT" ]]; then
        echo "    WARNING: No /dev/ttyACM* detected, but uploading anyway (Teensy uses HID)."
    else
        echo "    Detected: $PORT"
    fi

    FW_HEX="${FW_CACHE_DIR}/worker_${wid}.hex"
    echo "[Worker ${wid}] Uploading ${FW_HEX}..."

    # Copy cached hex back to build dir so PlatformIO uploads the correct firmware
    cp "$FW_HEX" "${BUILD_DIR}/firmware.hex"

    # Teensy upload uses teensy_loader_cli via USB HID, not serial port.
    # When only one Teensy is connected, PlatformIO auto-detects it.
    # You may need to press the button on the Teensy to enter bootloader mode.
    export EXTRA_BUILD_FLAGS="-DWORKER_ID=${wid}"  # needed for pio to resolve env
    if pio run -e "$PIO_ENV" -t upload 2>&1 | tail -5; then
        echo "[Worker ${wid}] ✓ Upload command finished"
        SUCCESS+=("$wid")
    else
        echo "[Worker ${wid}] ✗ Upload FAILED"
        FAILED+=("$wid")
    fi

    # If there are more workers to flash, prompt to swap
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
        echo "    Done with Worker ${wid}."
        echo "    Remaining: ${REMAINING[*]}"
        read -rp "    Unplug this Teensy, plug in the next one, then press ENTER... "
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