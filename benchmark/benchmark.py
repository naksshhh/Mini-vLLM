import asyncio
import aiohttp
import time
import statistics
import argparse
from typing import List

PROMPT = "The secret to building a high performance system is"

async def fetch_generation(session: aiohttp.ClientSession, url: str, max_tokens: int) -> float:
    start = time.time()
    async with session.post(url, json={"prompt": PROMPT, "max_new_tokens": max_tokens}) as response:
        await response.json()
        return time.time() - start

async def run_benchmark(concurrency: int, total_requests: int, url: str, max_tokens: int):
    print(f"Starting benchmark: {total_requests} requests with concurrency {concurrency}...")
    
    async with aiohttp.ClientSession() as session:
        queue = asyncio.Queue()
        for _ in range(total_requests):
            queue.put_nowait(1)
            
        latencies: List[float] = []
        
        async def worker():
            while not queue.empty():
                try:
                    await queue.get()
                    latency = await fetch_generation(session, url, max_tokens)
                    latencies.append(latency)
                except Exception as e:
                    print(f"Request failed: {e}")
                finally:
                    queue.task_done()
                    
        start_time = time.time()
        
        # Start concurrent workers
        workers = [asyncio.create_task(worker()) for _ in range(concurrency)]
        await asyncio.gather(*workers)
        
        total_time = time.time() - start_time
        
    if not latencies:
        print("All requests failed!")
        return
        
    p50 = statistics.median(latencies)
    p95 = statistics.quantiles(latencies, n=20)[18]  # 95th percentile
    
    # Calculate throughput (Requests / Second)
    # And total tokens / second
    throughput_reqs = total_requests / total_time
    throughput_tokens = (total_requests * max_tokens) / total_time
    
    print("\n--- Benchmark Results ---")
    print(f"Total Time:      {total_time:.2f}s")
    print(f"Throughput:      {throughput_reqs:.2f} requests/sec")
    print(f"Token Rate:      {throughput_tokens:.2f} tokens/sec")
    print(f"p50 Latency:     {p50*1000:.2f} ms")
    print(f"p95 Latency:     {p95*1000:.2f} ms")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM Inference Benchmark")
    parser.add_argument("--concurrency", type=int, default=10, help="Number of concurrent requests")
    parser.add_argument("--requests", type=int, default=50, help="Total number of requests")
    parser.add_argument("--url", type=str, default="http://localhost:8000/generate", help="API Endpoint")
    parser.add_argument("--max_tokens", type=int, default=50, help="Max tokens to generate per request")
    
    args = parser.parse_args()
    asyncio.run(run_benchmark(args.concurrency, args.requests, args.url, args.max_tokens))
