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

---

## Architecture

```
         User Requests
               │
               ▼
         FastAPI Gateway          ← /generate, /batch_generate, /health, /metrics
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
```

---

## Project Structure

```
ml-inference-system/
├── server/
│   ├── app.py          # FastAPI routes + Prometheus instrumentation
│   ├── model.py        # HF Transformers engine with KV-cache loop
│   ├── batching.py     # Async queue + background batch worker
│   └── kv_cache.py     # past_key_values cache manager
├── monitoring/
│   └── prometheus.yml  # Scrape config (model_server:8000/metrics)
├── load_test/
│   └── locustfile.py   # Locust: /generate (3x weight) + /batch_generate (1x)
├── benchmark/
│   └── benchmark.py    # Async p50/p95 latency + throughput calculator
├── docker/
│   ├── Dockerfile      # python:3.10-slim, pre-bakes model weights
│   └── docker-compose.yml  # model_server + prometheus + grafana
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
| `GET` | `/health` | Liveness check — returns model status |
| `POST` | `/generate` | Single prompt → text (routed via dynamic batcher) |
| `POST` | `/batch_generate` | List of prompts → list of texts |
| `GET` | `/metrics` | Prometheus scrape endpoint |

**Example:**
```bash
# Single generation
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Once upon a time", "max_new_tokens": 40}'

# Batch generation
curl -X POST http://localhost:8000/batch_generate \
  -H "Content-Type: application/json" \
  -d '{"prompts": ["What is AI?", "Tell me a joke"], "max_new_tokens": 30}'
```

---

## Benchmarking

```bash
# Run the built-in async benchmark
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

## Key Concepts Demonstrated

### Dynamic Batching
Instead of processing each HTTP request immediately (which wastes compute on small inputs), incoming requests are placed on an `asyncio.Queue`. A background worker collects them for up to **20 ms** or until **8 requests** accumulate, then runs a single batched model forward pass and fans results back to each request's `Future`.

### KV-Cache Optimization
Standard `.generate()` re-processes the entire prompt+history at every decode step (O(N²) attention cost). This system implements a manual decode loop:
1. **Prefill phase** — process full prompt once, save `past_key_values`
2. **Decode phase** — pass only the newest token + cached KVs each step → O(1) attention overhead per token

This mirrors the core optimization in vLLM and TensorRT-LLM.

### Observability
Three Prometheus counters/histograms are exposed:
- `inference_requests_total` — total request count
- `inference_tokens_generated_total` — total tokens output
- `inference_request_latency_seconds` — histogram with buckets at 0.1s → 50s

---

## Changing the Model

Set `MODEL_NAME` environment variable before starting:

```bash
# Locally
MODEL_NAME=gpt2 python -m uvicorn server.app:app --port 8000

# In Docker
# Edit docker/docker-compose.yml → environment → MODEL_NAME=<your-model>
```

Compatible with any HuggingFace CausalLM (e.g., `gpt2`, `distilgpt2`, `TinyLlama/TinyLlama-1.1B-Chat-v1.0`).
