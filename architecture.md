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
         │   ┌──────────────┐   ┌──────────────────┐   │
         │   │ Prefill Phase│   │   Decode Phase   │   │
         │   │ Process full │──▶│ 1 token at a     │  │
         │   │ prompt once  │   │ time using KV    │   │
         │   └──────────────┘   │ Cache            │   │
         │                      └──────────────────┘   │
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


| Service | Port | Purpose |
|---|---|---|
| `model_server` | 8000 | FastAPI + gpt2 inference |
| `prometheus` | 9090 | Metrics store |
| `grafana` | 3000 | Dashboard (admin/admin) |
