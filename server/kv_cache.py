import torch
from typing import Tuple, Optional

class KVCache:
    def __init__(self):
        self.past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None
        
    def update(self, past_key_values: Tuple[Tuple[torch.Tensor]]):
        """Update the cache with new key-value tensors."""
        self.past_key_values = past_key_values
        
    def get(self) -> Optional[Tuple[Tuple[torch.Tensor]]]:
        """Retrieve the current cached key-value tensors."""
        return self.past_key_values
        
    def clear(self):
        """Clear the cache."""
        self.past_key_values = None
