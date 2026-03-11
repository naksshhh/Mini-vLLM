#!/usr/bin/env bash
# tools/profile.sh
# Linux OS profiling script for the ML inference server.
# Captures: perf stat CPU counters, /proc memory metrics, taskset CPU-pinning comparison.
# Usage: ./tools/profile.sh [server_pid] [num_requests]
# Requires: perf, taskset (linux-tools-common), Python 3 with aiohttp

set -euo pipefail

SERVER_PID=${1:-$(pgrep -f "uvicorn server.app" | head -1)}
NUM_REQUESTS=${2:-50}
MAX_TOKENS=30
CONCURRENCY=5
SERVER_URL="http://localhost:8000"
REPORT_DIR="./profiling_reports"

mkdir -p "$REPORT_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
REPORT="$REPORT_DIR/profile_${TIMESTAMP}.txt"

echo "============================================"  | tee "$REPORT"
echo " ML Inference System — Linux OS Profile"     | tee -a "$REPORT"
echo " PID: $SERVER_PID | $(date)"                 | tee -a "$REPORT"
echo "============================================"  | tee -a "$REPORT"
echo ""                                             | tee -a "$REPORT"

# ─── 1. /proc memory snapshot ────────────────────────────────────────────────
echo "=== /proc/${SERVER_PID}/status — Memory Snapshot ===" | tee -a "$REPORT"
if [ -f "/proc/${SERVER_PID}/status" ]; then
    grep -E "^(VmPeak|VmRSS|VmSize|VmSwap|Threads|voluntary_ctxt_switches|nonvoluntary_ctxt_switches)" \
         /proc/${SERVER_PID}/status | tee -a "$REPORT"
else
    echo "  [WARN] /proc/${SERVER_PID}/status not found. Run on Linux." | tee -a "$REPORT"
fi
echo "" | tee -a "$REPORT"

# ─── 2. CPU affinity before pinning ──────────────────────────────────────────
echo "=== CPU Affinity (current) ===" | tee -a "$REPORT"
if command -v taskset &>/dev/null; then
    taskset -p $SERVER_PID | tee -a "$REPORT"
else
    echo "  [WARN] taskset not available. Install linux-tools-common." | tee -a "$REPORT"
fi
echo "" | tee -a "$REPORT"

# ─── 3. perf stat during a benchmark run ─────────────────────────────────────
echo "=== perf stat — CPU Performance Counters ===" | tee -a "$REPORT"
echo "  Running $NUM_REQUESTS requests with concurrency $CONCURRENCY..." | tee -a "$REPORT"

if command -v perf &>/dev/null; then
    perf stat -p $SERVER_PID \
        --interval-print 1000 \
        -e cycles,instructions,cache-misses,cache-references,branch-misses,context-switches \
        -- sleep 0 &
    PERF_PID=$!

    # Fire the benchmark while perf is attached
    python3 benchmark/benchmark.py \
        --concurrency $CONCURRENCY \
        --requests $NUM_REQUESTS \
        --max_tokens $MAX_TOKENS \
        --url "${SERVER_URL}/generate" 2>&1 | tee -a "$REPORT"

    wait $PERF_PID 2>&1 | tee -a "$REPORT"
else
    echo "  [WARN] perf not available. Run: sudo apt install linux-tools-common linux-tools-generic" | tee -a "$REPORT"
    # Still run benchmark without perf
    python3 benchmark/benchmark.py \
        --concurrency $CONCURRENCY \
        --requests $NUM_REQUESTS \
        --max_tokens $MAX_TOKENS \
        --url "${SERVER_URL}/generate" 2>&1 | tee -a "$REPORT"
fi
echo "" | tee -a "$REPORT"

# ─── 4. Post-run /proc memory snapshot ───────────────────────────────────────
echo "=== /proc/${SERVER_PID}/status — Memory After Load ===" | tee -a "$REPORT"
if [ -f "/proc/${SERVER_PID}/status" ]; then
    grep -E "^(VmPeak|VmRSS|VmSize|VmSwap|voluntary_ctxt_switches|nonvoluntary_ctxt_switches)" \
         /proc/${SERVER_PID}/status | tee -a "$REPORT"
fi
echo "" | tee -a "$REPORT"

# ─── 5. CPU pinning comparison ───────────────────────────────────────────────
echo "=== CPU Pinning Comparison (taskset) ===" | tee -a "$REPORT"
if command -v taskset &>/dev/null && [ -n "$SERVER_PID" ]; then
    echo "  [Baseline] Running benchmark without CPU pinning..." | tee -a "$REPORT"
    python3 benchmark/benchmark.py \
        --concurrency $CONCURRENCY \
        --requests $NUM_REQUESTS \
        --max_tokens $MAX_TOKENS \
        --url "${SERVER_URL}/generate" 2>&1 | tee -a "$REPORT"

    echo "" | tee -a "$REPORT"
    echo "  [Pinned] Pinning server PID $SERVER_PID to CPU cores 0-1..." | tee -a "$REPORT"
    taskset -cp 0-1 $SERVER_PID
    sleep 1

    python3 benchmark/benchmark.py \
        --concurrency $CONCURRENCY \
        --requests $NUM_REQUESTS \
        --max_tokens $MAX_TOKENS \
        --url "${SERVER_URL}/generate" 2>&1 | tee -a "$REPORT"

    echo "  [Restoring] Resetting CPU affinity to all cores..." | tee -a "$REPORT"
    taskset -cp 0-$(nproc --all | xargs -I{} expr {} - 1) $SERVER_PID
else
    echo "  [SKIP] taskset not available on this OS." | tee -a "$REPORT"
fi
echo "" | tee -a "$REPORT"

echo "============================================" | tee -a "$REPORT"
echo " Profile complete. Report saved to: $REPORT" | tee -a "$REPORT"
echo "============================================" | tee -a "$REPORT"
