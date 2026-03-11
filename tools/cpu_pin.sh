#!/usr/bin/env bash
# tools/cpu_pin.sh
# Benchmark the inference server with vs without CPU pinning using taskset.
# Demonstrates that pinning reduces context-switch overhead and cache thrashing.
#
# Usage (Linux only):
#   ./tools/cpu_pin.sh [server_pid] [num_requests] [concurrency]
#
# Prerequisites:
#   sudo apt install util-linux   # provides taskset
#   pip install aiohttp           # needed by benchmark.py

set -euo pipefail

SERVER_PID=${1:-$(pgrep -f "uvicorn server.app" | head -1)}
NUM_REQUESTS=${2:-50}
CONCURRENCY=${3:-5}
MAX_TOKENS=30
SERVER_URL="http://localhost:8000/generate"
TOTAL_CORES=$(nproc --all)

echo "======================================================="
echo "  CPU Pinning Benchmark  |  PID: $SERVER_PID"
echo "  Total CPU cores available: $TOTAL_CORES"
echo "======================================================="

if ! command -v taskset &>/dev/null; then
    echo "[ERROR] taskset not found. Install with: sudo apt install util-linux"
    exit 1
fi

# ── Baseline: all cores ────────────────────────────────────────────────────────
echo ""
echo "[BASELINE] Running without CPU pinning (all $TOTAL_CORES cores available)..."
taskset -cp "0-$((TOTAL_CORES-1))" $SERVER_PID > /dev/null 2>&1 || true
sleep 0.5

python3 benchmark/benchmark.py \
    --concurrency $CONCURRENCY \
    --requests $NUM_REQUESTS \
    --max_tokens $MAX_TOKENS \
    --url $SERVER_URL

# ── Restricted: 2 cores only ───────────────────────────────────────────────────
echo ""
echo "[PINNED-2] Restricting server to CPU cores 0-1 (2 cores)..."
taskset -cp 0-1 $SERVER_PID
sleep 0.5

python3 benchmark/benchmark.py \
    --concurrency $CONCURRENCY \
    --requests $NUM_REQUESTS \
    --max_tokens $MAX_TOKENS \
    --url $SERVER_URL

# ── Restore ────────────────────────────────────────────────────────────────────
echo ""
echo "[RESTORE] Restoring CPU affinity to all $TOTAL_CORES cores..."
taskset -cp "0-$((TOTAL_CORES-1))" $SERVER_PID > /dev/null 2>&1 || true

echo ""
echo "======================================================="
echo "  Note: Lower latency in PINNED-2 indicates CPU cache locality benefits."
echo "  Higher latency indicates insufficient parallelism on 2 cores."
echo "======================================================="
