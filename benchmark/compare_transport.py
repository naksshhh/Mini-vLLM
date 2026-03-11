"""
benchmark/compare_transport.py
Head-to-head HTTP vs gRPC latency and throughput comparison.
Sends N concurrent requests to both endpoints with identical payloads
and prints a side-by-side comparison.

Usage:
    # Start the HTTP server first:
    python -m uvicorn server.app:app --port 8000

    # Start the gRPC server concurrently:
    python -m server.grpc_server &

    # Run the comparison:
    python benchmark/compare_transport.py --concurrency 10 --requests 50 --max_tokens 30
"""

import asyncio
import argparse
import statistics
import sys
import time
from pathlib import Path

import aiohttp

# Try to import gRPC; if missing, warn but still run HTTP benchmark
try:
    import grpc
    sys.path.insert(0, str(Path(__file__).parent.parent / "server"))
    import inference_pb2
    import inference_pb2_grpc
    GRPC_AVAILABLE = True
except ImportError:
    GRPC_AVAILABLE = False
    print("[WARN] grpcio not installed or proto stubs missing. Only HTTP benchmark will run.")

PROMPT = "The future of distributed computing in data centers is"


# ── HTTP Benchmark ─────────────────────────────────────────────────────────────

async def http_request(session: aiohttp.ClientSession, url: str, max_tokens: int) -> float:
    start = time.perf_counter()
    async with session.post(url, json={"prompt": PROMPT, "max_new_tokens": max_tokens}) as resp:
        await resp.json()
    return (time.perf_counter() - start) * 1000  # ms


async def http_benchmark(http_url: str, concurrency: int, total: int, max_tokens: int) -> list[float]:
    latencies = []
    queue: asyncio.Queue = asyncio.Queue()
    for _ in range(total):
        queue.put_nowait(1)

    async with aiohttp.ClientSession() as session:
        async def worker():
            while not queue.empty():
                try:
                    queue.get_nowait()
                    lat = await http_request(session, http_url, max_tokens)
                    latencies.append(lat)
                except Exception as e:
                    print(f"  [HTTP error] {e}")

        await asyncio.gather(*[worker() for _ in range(concurrency)])

    return latencies


# ── gRPC Benchmark ─────────────────────────────────────────────────────────────

async def grpc_request(stub, max_tokens: int) -> float:
    start = time.perf_counter()
    await stub.Generate(inference_pb2.GenerateRequest(prompt=PROMPT, max_new_tokens=max_tokens))
    return (time.perf_counter() - start) * 1000  # ms


async def grpc_benchmark(grpc_address: str, concurrency: int, total: int, max_tokens: int) -> list[float]:
    if not GRPC_AVAILABLE:
        return []

    latencies = []
    queue: asyncio.Queue = asyncio.Queue()
    for _ in range(total):
        queue.put_nowait(1)

    async with grpc.aio.insecure_channel(grpc_address) as channel:
        stub = inference_pb2_grpc.InferenceServiceStub(channel)

        async def worker():
            while not queue.empty():
                try:
                    queue.get_nowait()
                    lat = await grpc_request(stub, max_tokens)
                    latencies.append(lat)
                except Exception as e:
                    print(f"  [gRPC error] {e}")

        await asyncio.gather(*[worker() for _ in range(concurrency)])

    return latencies


# ── Results Printer ────────────────────────────────────────────────────────────

def print_results(label: str, latencies: list[float], total: int, elapsed: float):
    if not latencies:
        print(f"\n  {label}: No successful requests.")
        return

    p50 = statistics.median(latencies)
    p95 = statistics.quantiles(latencies, n=20)[18]
    throughput = total / elapsed
    print(f"\n{'─'*45}")
    print(f"  {label}")
    print(f"{'─'*45}")
    print(f"  Successful Requests: {len(latencies)} / {total}")
    print(f"  Total Time:          {elapsed:.2f} s")
    print(f"  Throughput:          {throughput:.2f} req/s")
    print(f"  p50 Latency:         {p50:.2f} ms")
    print(f"  p95 Latency:         {p95:.2f} ms")
    print(f"  Min Latency:         {min(latencies):.2f} ms")
    print(f"  Max Latency:         {max(latencies):.2f} ms")


async def main():
    parser = argparse.ArgumentParser(description="HTTP vs gRPC transport comparison benchmark")
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--requests", type=int, default=50)
    parser.add_argument("--max_tokens", type=int, default=30)
    parser.add_argument("--http_url", default="http://localhost:8000/generate")
    parser.add_argument("--grpc_address", default="localhost:50051")
    args = parser.parse_args()

    print("=" * 45)
    print("  HTTP vs gRPC Transport Comparison")
    print(f"  Concurrency={args.concurrency}  Requests={args.requests}  max_tokens={args.max_tokens}")
    print("=" * 45)

    # ── HTTP ──
    print(f"\n[HTTP] Running {args.requests} requests against {args.http_url}...")
    t0 = time.perf_counter()
    http_lats = await http_benchmark(args.http_url, args.concurrency, args.requests, args.max_tokens)
    http_elapsed = time.perf_counter() - t0
    print_results("HTTP/1.1 + JSON (FastAPI)", http_lats, args.requests, http_elapsed)

    # ── gRPC ──
    if GRPC_AVAILABLE:
        print(f"\n[gRPC] Running {args.requests} requests against {args.grpc_address}...")
        t0 = time.perf_counter()
        grpc_lats = await grpc_benchmark(args.grpc_address, args.concurrency, args.requests, args.max_tokens)
        grpc_elapsed = time.perf_counter() - t0
        print_results("gRPC (HTTP/2 + Protobuf)", grpc_lats, args.requests, grpc_elapsed)

        # ── Comparison ──
        if http_lats and grpc_lats:
            http_p50 = statistics.median(http_lats)
            grpc_p50 = statistics.median(grpc_lats)
            savings = ((http_p50 - grpc_p50) / http_p50) * 100
            print(f"\n{'═'*45}")
            print(f"  Transport Overhead Comparison")
            print(f"{'═'*45}")
            print(f"  p50 latency reduction (gRPC vs HTTP): {savings:+.1f}%")
            print(f"  (Negative means gRPC is faster; reflects binary serialization")
            print(f"   and HTTP/2 multiplexing advantages at high concurrency)")
    else:
        print("\n[gRPC] Skipped — grpcio not available.")

    print()


if __name__ == "__main__":
    asyncio.run(main())
