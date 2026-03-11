import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Optional

from .kv_cache import KVCache

MODEL_NAME = os.getenv("MODEL_NAME", "gpt2")

class InferenceEngine:
    def __init__(self, model_name: str = MODEL_NAME):
        self.model_name = model_name
        print(f"Loading tokenizer for {self.model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Ensure padding token is set for batching
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        print(f"Loading model {self.model_name}...")
        # Use GPU if available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None
        )
        if self.device == "cpu":
            self.model.to(self.device)
            
        self.model.eval()
        print(f"Model loaded and ready on {self.device}!")

    def generate(self, prompt: str, max_new_tokens: int = 50) -> str:
        """Run standard inference for a single prompt using KV caching."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        
        cache = KVCache()
        generated_tokens = []
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                if cache.get() is not None:
                    # After the first step, input_ids is just the last generated token
                    model_inputs = {
                        "input_ids": input_ids[:, -1:],
                        "attention_mask": attention_mask,
                        "past_key_values": cache.get(),
                        "use_cache": True
                    }
                else:
                    # First step (Prefill phase): Process the whole prompt
                    model_inputs = {
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "use_cache": True
                    }

                outputs = self.model(**model_inputs)
                
                # Get the next token prediction
                next_token_logits = outputs.logits[:, -1, :]
                next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
                
                # Stop if EOS token
                if next_token_id.item() == self.tokenizer.eos_token_id:
                    break
                    
                generated_tokens.append(next_token_id.item())
                input_ids = torch.cat([input_ids, next_token_id], dim=-1)
                
                # Update attention mask for the new token
                attention_mask = torch.cat(
                    [attention_mask, torch.ones((1, 1), device=self.device, dtype=attention_mask.dtype)], 
                    dim=-1
                )
                
                # Update KV cache for the next iteration (Decode phase)
                cache.update(outputs.past_key_values)
                
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return generated_text
        
    def generate_batch(self, prompts: List[str], max_new_tokens: int = 50) -> List[str]:
        """Run batched inference for a list of prompts using KV caching."""
        inputs = self.tokenizer(
            prompts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True
        ).to(self.device)
        
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        batch_size = input_ids.shape[0]
        
        cache = KVCache()
        # Track generated tokens for each sequence in the batch
        generated_tokens = [[] for _ in range(batch_size)]
        
        # Track which sequences have hit EOS
        finished_sequences = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                if cache.get() is not None:
                    model_inputs = {
                        "input_ids": input_ids[:, -1:],
                        "attention_mask": attention_mask,
                        "past_key_values": cache.get(),
                        "use_cache": True
                    }
                else:
                    model_inputs = {
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "use_cache": True
                    }

                outputs = self.model(**model_inputs)
                
                next_token_logits = outputs.logits[:, -1, :]
                next_token_ids = torch.argmax(next_token_logits, dim=-1)
                
                # If a sequence is already finished, force its next token to be pad_token
                next_token_ids = torch.where(
                    finished_sequences,
                    torch.tensor(self.tokenizer.pad_token_id, device=self.device),
                    next_token_ids
                )
                
                # Update finished status
                is_eos = next_token_ids == self.tokenizer.eos_token_id
                finished_sequences = finished_sequences | is_eos
                
                # Append predicted tokens to our tracking lists
                for i in range(batch_size):
                    if not finished_sequences[i] or is_eos[i]:
                        # Only add if it's not finished, or this is the exact step it finishes
                        generated_tokens[i].append(next_token_ids[i].item())
                
                # Stop early if all sequences are finished
                if finished_sequences.all():
                    break
                    
                input_ids = torch.cat([input_ids, next_token_ids.unsqueeze(-1)], dim=-1)
                attention_mask = torch.cat(
                    [attention_mask, torch.ones((batch_size, 1), device=self.device, dtype=attention_mask.dtype)], 
                    dim=-1
                )
                cache.update(outputs.past_key_values)
                
        # Decode ignoring pad tokens we might have forced
        generated_texts = [
            self.tokenizer.decode([t for t in tokens if t != self.tokenizer.pad_token_id], skip_special_tokens=True) 
            for tokens in generated_tokens
        ]
        return generated_texts

engine = None

def get_engine():
    global engine
    if engine is None:
        engine = InferenceEngine()
    return engine
