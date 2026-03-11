from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import time
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response

from .model import get_engine
from .batching import DynamicBatcher

app = FastAPI(title="ML Inference System", description="High-Performance LLM Inference Server")

class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 50

class GenerateResponse(BaseModel):
    text: str
    latency_ms: float

class BatchGenerateRequest(BaseModel):
    prompts: List[str]
    max_new_tokens: int = 50

class BatchGenerateResponse(BaseModel):
    texts: List[str]
    latency_ms: float

# Prometheus Metrics
REQUEST_COUNT = Counter("inference_requests_total", "Total number of requests")
TOKEN_COUNT = Counter("inference_tokens_generated_total", "Total number of tokens generated")
REQUEST_LATENCY = Histogram(
    "inference_request_latency_seconds", 
    "Request latency in seconds",
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
)

# Global batcher instance
batcher: DynamicBatcher = None

@app.on_event("startup")
async def startup_event():
    global batcher
    # Preload the model on startup so the first request isn't slow
    engine = get_engine()
    
    # Initialize the batcher with the engine's batch generation function
    # Wait max 20ms and group up to 8 requests
    batcher = DynamicBatcher(
        engine_generate_batch_fn=engine.generate_batch,
        max_batch_size=8,
        timeout_ms=20
    )
    batcher.start()

@app.on_event("shutdown")
async def shutdown_event():
    global batcher
    if batcher:
        await batcher.stop()

@app.get("/health")
async def health_check():
    return {"status": "ok", "model_loaded": get_engine() is not None}

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    REQUEST_COUNT.inc()
    start_time = time.time()
    
    try:
        global batcher
        # The batcher returns both the result text and the latency measured 
        # from time of enqueueing to time of completion.
        result, latency = await batcher.process_request(
            req.prompt, 
            req.max_new_tokens
        )
        
        # Approximate tokens based on text splitting for simplicity, 
        # though ideally we'd count actual tokenizer output length.
        TOKEN_COUNT.inc(len(result.split()))
        REQUEST_LATENCY.observe(time.time() - start_time)
        
        return GenerateResponse(
            text=result,
            latency_ms=latency
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch_generate", response_model=BatchGenerateResponse)
async def batch_generate(req: BatchGenerateRequest):
    try:
        start_time = time.time()
        
        engine = get_engine()
        results = engine.generate_batch(req.prompts, req.max_new_tokens)
        
        end_time = time.time()
        
        return BatchGenerateResponse(
            texts=results,
            latency_ms=(end_time - start_time) * 1000
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=8000, reload=True)
