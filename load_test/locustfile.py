from locust import HttpUser, task, between
import random

PROMPTS = [
    "The meaning of life is",
    "Once upon a time in a faraway land",
    "Here is a short story about a robot:",
    "To build a high performance system, you must",
    "In the year 2050, artificial intelligence will",
]

class MLInferenceUser(HttpUser):
    # Wait time between tasks: 1 to 5 seconds
    wait_time = between(1.0, 5.0)

    @task(3)
    def generate_single(self):
        """Simulate a user hitting the single inference endpoint."""
        prompt = random.choice(PROMPTS)
        self.client.post("/generate", json={
            "prompt": prompt,
            "max_new_tokens": random.randint(20, 50)
        })

    @task(1)
    def generate_batch(self):
        """Simulate a user submitting a batch of prompts."""
        prompts = random.choices(PROMPTS, k=random.randint(2, 4))
        self.client.post("/batch_generate", json={
            "prompts": prompts,
            "max_new_tokens": random.randint(20, 50)
        })
