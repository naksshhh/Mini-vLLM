# High-Performance LLM Inference System

A production-ready LLM inference API demonstrating core ML systems engineering: **dynamic batching**, **KV-cache optimization**, and **Prometheus/Grafana observability** — all containerized with Docker.

> **Benchmark Results** (gpt2, 10 concurrent workers, 30 tokens/request, CPU):
> ```
> Throughput:   6.75 req/s   |   Token Rate: 202 tokens/s
> p50 Latency:  1142 ms      |   p95 Latency: 2415 ms
> ```

---

## What This Demonstrates

- ✅ **Dynamic Batching Scheduler** — async queue groups up to 8 requests per 20ms window, boosting GPU/CPU utilization
- ✅ **KV-Cache Optimization** — manual token-generation loop using `past_key_values` to skip prompt recomputation each decode step (O(1) per token vs O(N²))
- ✅ **Prometheus Metrics** — tracks `request_count`, `token_count`, and `latency` histograms (p50/p90/p95)
- ✅ **Grafana Dashboard** — live time-series panels for throughput, latency percentiles, and request rates
- ✅ **Docker Compose** — one command to bring up API server + Prometheus + Grafana
- ✅ **Locust Load Testing** — simulates 100+ concurrent users across `/generate` and `/batch_generate`
- ✅ **gRPC Transport** — binary protobuf serialization + HTTP/2 multiplexing alongside FastAPI with head-to-head comparison benchmark
- ✅ **Linux OS Profiling** — `/proc` memory tracking, `perf stat` CPU counters, `taskset` CPU pinning tools
- ✅ **Distributed Multi-Worker** — async round-robin router with health checking, automatic failover, and per-worker stats

---

## Architecture

```
         User Requests
               │
               ▼
         FastAPI Gateway          ← /generate, /batch_generate, /health, /metrics, /sys/info
               │
       Async Request Queue        ← asyncio.Queue (max queue size, backpressure)
               │
       Dynamic Batch Builder      ← waits 20ms or max_batch_size=8
               │
       Inference Engine           ← HuggingFace Transformers (gpt2 / switchable)
          │         │
    KV Cache     Model.forward()  ← prefill once, decode with cached past_key_values
               │
    Prometheus /metrics           ← scraped every 5s
               │
        Grafana Dashboard         ← latency p50/p95, throughput, token rate

══════ gRPC Transport (parallel to HTTP) ══════
    InferenceServicer (grpc_server.py)
    ── same DynamicBatcher ──▶ same InferenceEngine
    protobuf binary encoding over HTTP/2

══════ Distributed Mode ══════
    Router :8080  ──round-robin──▶  Worker-1 :8001
                  ──round-robin──▶  Worker-2 :8002
    Per-worker health check every 5s, auto-failover
```

---

## Project Structure

```
ml-inference-system/
├── server/
│   ├── app.py              # FastAPI routes + Prometheus + /sys/info OS metrics
│   ├── model.py            # HF Transformers engine with KV-cache loop
│   ├── batching.py         # Async queue + background batch worker
│   ├── kv_cache.py         # past_key_values cache manager
│   ├── grpc_server.py      # gRPC servicer (same batcher/engine as HTTP)
│   ├── inference_pb2.py    # Generated protobuf classes
│   └── inference_pb2_grpc.py  # Generated gRPC stubs
├── proto/
│   └── inference.proto     # Service definition (Generate, BatchGenerate, Health)
├── distributed/
│   ├── worker.py           # Stateless inference worker node
│   ├── router.py           # Async round-robin router with health checking
│   └── run_cluster.py      # Launch N workers + router locally
├── tools/
│   ├── profile.sh          # perf stat + /proc memory + taskset pinning (Linux)
│   ├── monitor_proc.py     # Live /proc/<pid>/status CSV logger
│   └── cpu_pin.sh          # CPU pinning benchmark comparison
├── monitoring/
│   └── prometheus.yml      # Scrape config
├── load_test/
│   └── locustfile.py       # Locust load simulator
├── benchmark/
│   ├── benchmark.py        # Async p50/p95 latency + throughput tool
│   └── compare_transport.py  # HTTP vs gRPC head-to-head comparison
├── docker/
│   ├── Dockerfile
│   ├── docker-compose.yml             # Single-node: server + prometheus + grafana
│   └── docker-compose-distributed.yml # Multi-node: router + 2 workers + prometheus + grafana
├── architecture.md
└── README.md
```

---

## Quick Start

### Local Development (no Docker)

```bash
# 1. Create virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1        # Windows
# source venv/bin/activate         # Linux/Mac

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start the server
python -m uvicorn server.app:app --host 0.0.0.0 --port 8000

# 4. Test it
curl -X POST http://localhost:8000/generate \
     -H "Content-Type: application/json" \
     -d '{"prompt": "The meaning of life is", "max_new_tokens": 50}'
```

### Full Stack with Docker (API + Prometheus + Grafana)

```bash
docker-compose -f docker/docker-compose.yml up --build -d
```

| Service | URL |
|---|---|
| API Docs (Swagger) | http://localhost:8000/docs |
| Raw Metrics | http://localhost:8000/metrics |
| OS Process Metrics | http://localhost:8000/sys/info |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 (admin / admin) |

> In Grafana: **Dashboards → LLM Inference System** (auto-provisioned on first run)

```bash
# Stop everything
docker-compose -f docker/docker-compose.yml down
```

---

## Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness check |
| `POST` | `/generate` | Single prompt → text (routed via dynamic batcher) |
| `POST` | `/batch_generate` | List of prompts → list of texts |
| `GET` | `/metrics` | Prometheus scrape endpoint |
| `GET` | `/sys/info` | Live OS metrics: RSS memory, threads, context switches |

---

## Benchmarking

```bash
# Standard latency + throughput benchmark
python benchmark/benchmark.py --concurrency 10 --requests 100 --max_tokens 30

# Sample output (gpt2, CPU, 10 concurrent workers):
# Throughput:    6.75 requests/sec
# Token Rate:   202.41 tokens/sec
# p50 Latency: 1142.11 ms
# p95 Latency: 2415.13 ms
```

```bash
# Run Locust load test (100 users, 10 spawn/s, 1 minute)
.\venv\Scripts\locust -f load_test/locustfile.py \
  --headless -u 100 -r 10 --run-time 60s \
  --host http://localhost:8000
```

---

## gRPC Transport

The system exposes both HTTP and gRPC interfaces backed by the same inference engine.

```bash
# Regenerate protobuf stubs (if you modify inference.proto)
python -m grpc_tools.protoc -I./proto --python_out=./server --grpc_python_out=./server ./proto/inference.proto

# Start the gRPC server (port 50051)
python -m server.grpc_server

# Run head-to-head HTTP vs gRPC comparison
python benchmark/compare_transport.py --concurrency 10 --requests 50 --max_tokens 30
```

**gRPC advantages at high concurrency:**
- Binary protobuf serialization (no JSON parsing overhead)
- HTTP/2 multiplexing: multiple requests over one TCP connection
- Lower per-message overhead at scale

---

## Linux OS Profiling (Linux / WSL2 / Docker shell)

```bash
# Get into the running container
docker exec -it docker-model_server-1 bash

# Profile with perf stat + /proc memory tracking
./tools/profile.sh <server_pid> 50

# Live /proc monitor (saves CSV to profiling_reports/)
python tools/monitor_proc.py --pid <server_pid> --duration 60

# CPU pinning comparison (requires taskset)
./tools/cpu_pin.sh <server_pid>

# View live OS metrics via API
curl http://localhost:8000/sys/info
```

What the profiling shows:
- **VmRSS / VmPeak** — actual physical memory footprint of the model
- **Voluntary context switches** — how often the server yields CPU (correlates with async efficiency)
- **IPC (instructions per cycle)** — compute density during batch inference
- **Cache miss rate** — impact of CPU pinning on memory locality

---

## Distributed Multi-Worker

Run a cluster of independent inference workers behind a round-robin router.

### Local (no Docker)
```bash
# Launch 2 workers + router (blocking, Ctrl+C to stop)
python distributed/run_cluster.py --workers 2 --base-port 8001 --router-port 8080

# Send requests to the router (note: worker_id in response shows which worker handled it)
curl -X POST http://localhost:8080/generate \
     -H "Content-Type: application/json" \
     -d '{"prompt": "Distributed inference means", "max_new_tokens": 30}'

# View per-worker stats
curl http://localhost:8080/stats
```

### Docker (multi-container)
```bash
# Start distributed cluster (router:8080, worker_1:8001, worker_2:8002)
docker-compose -f docker/docker-compose-distributed.yml up --build -d

# Benchmark the distributed router
python benchmark/benchmark.py --concurrency 20 --requests 100 --url http://localhost:8080/generate
```

---

## Key Concepts Demonstrated

### Dynamic Batching
Incoming requests wait up to **20ms** or until **8 requests** accumulate. A single batched model forward pass serves all — throughput scales near-linearly with batch size while latency cost is amortized.

### KV-Cache Optimization
Transformer attention recomputes over O(N²) tokens per decode step. This system implements a manual decode loop:
1. **Prefill** — full prompt processed once → `past_key_values` saved
2. **Decode** — only newest token + cached KVs passed per step → O(1) attention cost

This mirrors vLLM's core optimization (which extends it further with PagedAttention).

### gRPC vs HTTP
- **HTTP/1.1 + JSON**: human-readable, widely compatible, higher per-message overhead
- **gRPC (HTTP/2 + Protobuf)**: binary encoding, multiplexed connections, lower latency at scale
- Run `benchmark/compare_transport.py` to measure the difference under your workload.

### Distributed Routing
The router maintains a live health registry. Unhealthy workers (failed health checks) are removed from the round-robin pool and re-added when they recover. The `/stats` endpoint shows per-worker request counts and average latency — useful for detecting load imbalance.

### Linux OS Observability
- `/proc/self/status` — VmRSS (resident set size), thread count, context switches
- `perf stat` — IPC, cache miss rates, branch mispredictions during inference
- `taskset` — CPU pinning to restrict NUMA memory locality and measure cache effects

---

## Changing the Model

```bash
MODEL_NAME=gpt2 python -m uvicorn server.app:app --port 8000
```

Compatible with any HuggingFace CausalLM (e.g., `gpt2`, `distilgpt2`, `TinyLlama/TinyLlama-1.1B-Chat-v1.0`).

