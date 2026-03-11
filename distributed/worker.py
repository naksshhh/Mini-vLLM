"""
distributed/worker.py
A standalone stateless inference worker node.
Exposes a lightweight FastAPI server with /generate and /health.
Designed to be launched as multiple independent processes, each
serving requests from the distributed router (distributed/router.py).

Launch examples:
    python -m distributed.worker --port 8001
    python -m distributed.worker --port 8002

The router will discover these via its registry.
"""
import argparse
import asyncio
import os
import sys
import time

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Allow running as module from the project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from server.model import get_engine

app = FastAPI(title="ML Inference Worker")

# Worker identity (set at start)
WORKER_ID = os.getenv("WORKER_ID", "worker-0")
WORKER_PORT = int(os.getenv("WORKER_PORT", "8001"))


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 50


class GenerateResponse(BaseModel):
    text: str
    latency_ms: float
    worker_id: str  # Includes the worker identity so we can verify routing


@app.on_event("startup")
async def startup():
    """Preload model on worker startup."""
    print(f"[Worker {WORKER_ID}] Loading model...")
    get_engine()
    print(f"[Worker {WORKER_ID}] Ready on port {WORKER_PORT}")


@app.get("/health")
async def health():
    return {"status": "ok", "worker_id": WORKER_ID, "port": WORKER_PORT}


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    try:
        start = time.time()
        engine = get_engine()
        # Run CPU-bound inference in a thread so the event loop stays responsive
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None, engine.generate, req.prompt, req.max_new_tokens
        )
        latency = (time.time() - start) * 1000
        return GenerateResponse(text=result, latency_ms=latency, worker_id=WORKER_ID)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference Worker Node")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--worker-id", type=str, default=None)
    args = parser.parse_args()

    WORKER_PORT = args.port
    WORKER_ID = args.worker_id or f"worker-{args.port}"
    os.environ["WORKER_ID"] = WORKER_ID
    os.environ["WORKER_PORT"] = str(WORKER_PORT)

    uvicorn.run(app, host="0.0.0.0", port=WORKER_PORT)
