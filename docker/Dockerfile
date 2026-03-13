FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
# procps      → provides pgrep, ps, top (needed by tools/profile.sh)
# linux-perf  → perf stat for CPU counter profiling
# util-linux  → provides taskset for CPU pinning
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    procps \
    linux-perf \
    util-linux \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy ALL project code (not just server/)
COPY server/ /app/server/
COPY tools/ /app/tools/
COPY benchmark/ /app/benchmark/
COPY distributed/ /app/distributed/
COPY monitoring/ /app/monitoring/

# Make shell scripts executable
RUN chmod +x /app/tools/profile.sh /app/tools/cpu_pin.sh

# Pre-download the default model (gpt2) during build
ENV MODEL_NAME=gpt2
RUN python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; AutoTokenizer.from_pretrained('$MODEL_NAME'); AutoModelForCausalLM.from_pretrained('$MODEL_NAME')"

# Expose FastAPI port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
