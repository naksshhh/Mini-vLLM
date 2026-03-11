# Architecture Overview

This project implements a production-grade LLM Inference Server demonstrating ML systems engineering concepts: dynamic batching, KV-cache optimization, and observability.

## System Architecture

```
         ┌─────────────────────────────────────────────┐
         │              FastAPI Gateway                │
         │   /generate  /batch_generate  /health       │
         │   /metrics (Prometheus scrape endpoint)     │
         └──────────────────┬──────────────────────────┘
                            │  HTTP Request
                            ▼
         ┌─────────────────────────────────────────────┐
         │           Async Request Queue               │
         │    asyncio.Queue — max_size, backpressure   │
         └──────────────────┬──────────────────────────┘
                            │  RequestItem + Future
                            ▼
         ┌─────────────────────────────────────────────┐
         │         Dynamic Batch Builder               │
         │  Waits 20ms OR until max_batch_size=8       │
         │  Runs inference in thread pool executor     │
         │  Resolves each request's Future             │
         └──────────────────┬──────────────────────────┘
                            │  Batched tensor inputs
                            ▼
         ┌─────────────────────────────────────────────┐
         │           Inference Engine                  │
         │  HuggingFace AutoModelForCausalLM           │
         │  Runs on CPU (or GPU if available)          │
         │                                             │
         │   ┌──────────────┐   ┌──────────────────┐  │
         │   │ Prefill Phase│   │   Decode Phase   │  │
         │   │ Process full │──▶│ 1 token at a    │  │
         │   │ prompt once  │   │ time using KV   │  │
         │   └──────────────┘   │ Cache           │  │
         │                      └──────────────────┘  │
         └──────────────────┬──────────────────────────┘
                            │
              ┌─────────────┼──────────────┐
              ▼             ▼              ▼
     ┌──────────────┐ ┌──────────┐ ┌──────────────┐
     │  KV Cache    │ │  Model   │ │  Prometheus  │
     │  Manager     │ │ Weights  │ │  Metrics     │
     │past_key_vals │ │ (gpt2)   │ │  /metrics    │
     └──────────────┘ └──────────┘ └──────┬───────┘
                                          │ scrape: 5s
                                          ▼
                                   ┌────────────┐
                                   │ Prometheus │
                                   │ :9090      │
                                   └─────┬──────┘
                                         │ datasource
                                         ▼
                                   ┌────────────┐
                                   │  Grafana   │
                                   │  :3000     │
                                   │ dashboard  │
                                   └────────────┘
```

## Core Components

### 1. API Gateway (`server/app.py`)
FastAPI application that exposes REST endpoints. Each `/generate` request is converted into a `RequestItem` with an `asyncio.Future` and placed on the batch queue. The endpoint then `await`s the Future — it does not block any threads. Prometheus counters and histograms are incremented on each completed request.

### 2. Dynamic Batching Scheduler (`server/batching.py`)
The `DynamicBatcher` runs a background `asyncio.Task` (the worker loop) that:
1. Blocks on `queue.get()` until the first request arrives
2. Tries to collect more requests with `asyncio.wait_for(queue.get(), timeout=0.020)` (20ms)
3. Stops collecting at `max_batch_size=8` or after timeout
4. Dispatches the batch via `loop.run_in_executor()` so the model's CPU/GPU work doesn't block the event loop
5. Resolves each request's Future with its output text

**Why this matters:** Without batching, each request uses the hardware at ~10% capacity. With batching, multiple prompts are processed as a single matrix multiply — the primary operation is the same cost regardless of batch size, so throughput increases near-linearly.

### 3. Inference Engine (`server/model.py`)
Wraps `AutoModelForCausalLM` and `AutoTokenizer`. Uses FP16 on CUDA if available, FP32 on CPU. Implements two generation paths both using explicit KV caching.

### 4. KV Cache Optimization (`server/kv_cache.py`)
Transformer attention is O(N²) in sequence length. Standard `.generate()` re-runs the entire attention computation for prompt + previously generated tokens at each decode step.

This system implements a **manual decode loop**:

| Phase | Input | Cost |
|---|---|---|
| **Prefill** | Full prompt tokens | O(N²) — once |
| **Decode step 1** | 1 new token + cached KVs | O(N) |
| **Decode step 2** | 1 new token + cached KVs | O(N) |
| **...** | | O(N) each |

The `KVCache` class stores the `past_key_values` tensor tuple returned by `model.forward()` and feeds it back on the next step. This is the fundamental optimization behind vLLM's PagedAttention (which extends it with memory pooling across sequences).

### 5. Observability Stack

| Component | Role |
|---|---|
| `prometheus-client` in FastAPI | Exposes `/metrics` in Prometheus text format |
| `prometheus.yml` | Scraper config — polls `model_server:8000/metrics` every 5s |
| Grafana | Queries Prometheus, renders time-series: request rate, token throughput, p50/p95 latency |

## Benchmark Results

Real measurements on: Intel i5-12500H, 16GB RAM, RTX 3050 4GB (running on CPU fallback in Docker)

```
Concurrency: 10 workers | 100 requests | 30 max_new_tokens
─────────────────────────────────────────────────────────
Total Time:      14.82 s
Throughput:       6.75 requests/sec
Token Rate:     202.41 tokens/sec
p50 Latency:   1142.11 ms
p95 Latency:   2415.13 ms
```

## Docker Services

```
docker-compose -f docker/docker-compose.yml up --build -d
```

| Service | Port | Purpose |
|---|---|---|
| `model_server` | 8000 | FastAPI + gpt2 inference |
| `prometheus` | 9090 | Metrics store |
| `grafana` | 3000 | Dashboard (admin/admin) |
