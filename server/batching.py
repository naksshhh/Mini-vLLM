import asyncio
import time
from typing import List, Tuple

class RequestItem:
    def __init__(self, prompt: str, max_new_tokens: int):
        self.prompt = prompt
        self.max_new_tokens = max_new_tokens
        # Future to send the result back to the FastAPI handler
        self.future = asyncio.Future()
        self.enqueue_time = time.time()

class DynamicBatcher:
    def __init__(self, engine_generate_batch_fn, max_batch_size: int = 8, timeout_ms: int = 20):
        self.engine_generate_batch_fn = engine_generate_batch_fn
        self.max_batch_size = max_batch_size
        self.timeout_ms = timeout_ms / 1000.0  # seconds
        
        self.queue: asyncio.Queue[RequestItem] = asyncio.Queue()
        self.worker_task: asyncio.Task | None = None
        
    def start(self):
        if self.worker_task is None:
            self.worker_task = asyncio.create_task(self._worker_loop())
            
    async def stop(self):
        if self.worker_task is not None:
            self.worker_task.cancel()
            try:
                await self.worker_task
            except asyncio.CancelledError:
                pass
            self.worker_task = None
            
    async def process_request(self, prompt: str, max_new_tokens: int) -> Tuple[str, float]:
        """Called by the API endpoint to enqueue a request and wait for the result."""
        item = RequestItem(prompt, max_new_tokens)
        await self.queue.put(item)
        
        # Wait for the background worker to set the result
        result = await item.future
        latency = (time.time() - item.enqueue_time) * 1000
        return result, latency
        
    async def _worker_loop(self):
        """Background task that groups requests into batches."""
        while True:
            batch: List[RequestItem] = []
            
            try:
                # Wait for the first item with no timeout
                first_item = await self.queue.get()
                batch.append(first_item)
                
                # Now try to gather more items up to max_batch_size, 
                # but only wait for timeout_ms
                timeout_time = time.time() + self.timeout_ms
                
                while len(batch) < self.max_batch_size:
                    time_left = timeout_time - time.time()
                    if time_left <= 0:
                        break
                        
                    try:
                        # Wait for another item within the remaining time
                        item = await asyncio.wait_for(self.queue.get(), timeout=time_left)
                        batch.append(item)
                    except asyncio.TimeoutError:
                        break
                
                # We have a batch, let's process it
                await self._process_batch(batch)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Error in batch worker: {e}")
                
    async def _process_batch(self, batch: List[RequestItem]):
        """Run the batch through the inference engine and resolve the futures."""
        if not batch:
            return
            
        prompts = [item.prompt for item in batch]
        # In a real scenario we might need to group by max_new_tokens, 
        # but for simplicity we'll just take the max requested in the batch
        max_tokens = max(item.max_new_tokens for item in batch)
        
        try:
            # The actual InferenceEngine is synchronous and CPU-bound, 
            # so we should run it in an executor to not block the asyncio loop
            loop = asyncio.get_running_loop()
            results = await loop.run_in_executor(
                None, 
                self.engine_generate_batch_fn, 
                prompts, 
                max_tokens
            )
            
            # Resolve all futures with their respective results
            for item, result in zip(batch, results):
                if not item.future.done():
                    item.future.set_result(result)
                    
        except Exception as e:
            # If the batch fails, fail all futures
            for item in batch:
                if not item.future.done():
                    item.future.set_exception(e)
        finally:
            # Mark queue tasks as done
            for _ in batch:
                self.queue.task_done()
