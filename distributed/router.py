"""
distributed/router.py
Async round-robin load balancer for the ML inference cluster.
Maintains a registry of worker addresses, routes requests to workers in order,
and health-checks them periodically. Routes to the next healthy worker on failure.

Run:
    python -m distributed.router --workers http://localhost:8001 http://localhost:8002 --port 8080

Then send requests to the router at http://localhost:8080/generate
"""
import argparse
import asyncio
import os
import sys
import time
from itertools import cycle
from typing import List

import aiohttp
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="ML Inference Router")


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 50


class RouterState:
    """Holds the live worker registry and health status."""
    def __init__(self, worker_urls: List[str]):
        self.all_workers = worker_urls
        self.healthy_workers: List[str] = list(worker_urls)
        self._cycle = cycle(self.healthy_workers)
        self._lock = asyncio.Lock()
        # Per-worker stats
        self.stats = {url: {"requests": 0, "errors": 0, "total_latency_ms": 0.0}
                      for url in worker_urls}

    def next_worker(self) -> str:
        """Round-robin: return next healthy worker URL."""
        if not self.healthy_workers:
            raise RuntimeError("No healthy workers available")
        return next(self._cycle)

    def record(self, url: str, latency_ms: float):
        self.stats[url]["requests"] += 1
        self.stats[url]["total_latency_ms"] += latency_ms

    def record_error(self, url: str):
        self.stats[url]["errors"] += 1

    def mark_unhealthy(self, url: str):
        if url in self.healthy_workers:
            self.healthy_workers.remove(url)
            # Rebuild cycle from current healthy workers
            self._cycle = cycle(self.healthy_workers) if self.healthy_workers else iter([])
            print(f"[Router] Worker {url} marked UNHEALTHY. Active: {len(self.healthy_workers)}")

    def mark_healthy(self, url: str):
        if url not in self.healthy_workers:
            self.healthy_workers.append(url)
            self._cycle = cycle(self.healthy_workers)
            print(f"[Router] Worker {url} recovered. Active: {len(self.healthy_workers)}")


state: RouterState = None


async def health_check_loop(interval: int = 5):
    """Periodically checks each worker's /health endpoint."""
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=3)) as session:
        while True:
            for url in state.all_workers:
                try:
                    async with session.get(f"{url}/health") as resp:
                        if resp.status == 200:
                            state.mark_healthy(url)
                        else:
                            state.mark_unhealthy(url)
                except Exception:
                    state.mark_unhealthy(url)
            await asyncio.sleep(interval)


@app.on_event("startup")
async def startup():
    # Start background health checker
    asyncio.create_task(health_check_loop())
    print(f"[Router] Started. Workers: {state.all_workers}")


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "healthy_workers": state.healthy_workers,
        "all_workers": state.all_workers,
        "worker_stats": state.stats,
    }


@app.post("/generate")
async def generate(req: GenerateRequest):
    """Route to next available worker and return result."""
    if not state.healthy_workers:
        raise HTTPException(status_code=503, detail="No healthy workers available")

    worker_url = state.next_worker()
    start = time.time()

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60)) as session:
        try:
            async with session.post(
                f"{worker_url}/generate",
                json={"prompt": req.prompt, "max_new_tokens": req.max_new_tokens}
            ) as resp:
                if resp.status != 200:
                    state.record_error(worker_url)
                    raise HTTPException(status_code=resp.status, detail=await resp.text())
                result = await resp.json()
                latency = (time.time() - start) * 1000
                state.record(worker_url, latency)
                # Tag with router info
                result["routed_to"] = worker_url
                result["router_latency_ms"] = latency
                return result
        except aiohttp.ClientError as e:
            state.record_error(worker_url)
            state.mark_unhealthy(worker_url)
            raise HTTPException(status_code=503, detail=f"Worker {worker_url} unreachable: {e}")


@app.get("/stats")
async def stats():
    """Per-worker request statistics."""
    summary = {}
    for url, s in state.stats.items():
        avg_lat = (s["total_latency_ms"] / s["requests"]) if s["requests"] > 0 else 0
        summary[url] = {**s, "avg_latency_ms": round(avg_lat, 2)}
    return {"workers": summary, "healthy": state.healthy_workers}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ML Inference Router")
    parser.add_argument("--workers", nargs="+",
                        default=["http://localhost:8001", "http://localhost:8002"],
                        help="List of worker URLs")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    state = RouterState(args.workers)
    uvicorn.run(app, host="0.0.0.0", port=args.port)
