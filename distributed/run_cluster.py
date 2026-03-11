"""
distributed/run_cluster.py
Launches a local distributed inference cluster: N worker nodes + 1 router.
Each worker loads the model from scratch (simulates separate machine processes).
The router performs health-check polling and round-robin routing.

Usage:
    python distributed/run_cluster.py --workers 2 --base-port 8001 --router-port 8080

Then hit the router:
    curl -X POST http://localhost:8080/generate
         -H 'Content-Type: application/json'
         -d '{"prompt": "Hello, distributed world!", "max_new_tokens": 30}'

Check per-worker stats at:
    curl http://localhost:8080/stats
"""
import argparse
import subprocess
import sys
import time
import os
import signal

import requests


def wait_for_health(url: str, max_wait: int = 60) -> bool:
    """Poll /health until the server responds or timeout."""
    deadline = time.time() + max_wait
    while time.time() < deadline:
        try:
            resp = requests.get(f"{url}/health", timeout=2)
            if resp.status_code == 200:
                return True
        except requests.ConnectionError:
            pass
        time.sleep(1)
    return False


def launch_cluster(num_workers: int, base_port: int, router_port: int):
    procs = []
    worker_urls = []

    print(f"\n{'='*55}")
    print(f"  Launching ML Inference Cluster")
    print(f"  Workers: {num_workers}  |  Router: :{router_port}")
    print(f"{'='*55}\n")

    # ── Launch worker processes ──────────────────────────────────────────────
    for i in range(num_workers):
        port = base_port + i
        worker_id = f"worker-{i+1}"
        url = f"http://localhost:{port}"
        worker_urls.append(url)

        env = os.environ.copy()
        env["WORKER_ID"] = worker_id
        env["WORKER_PORT"] = str(port)

        print(f"[Cluster] Starting {worker_id} on port {port}...")
        proc = subprocess.Popen(
            [sys.executable, "-m", "distributed.worker", "--port", str(port), "--worker-id", worker_id],
            env=env
        )
        procs.append(proc)

    # ── Wait for workers to be healthy ───────────────────────────────────────
    print("\n[Cluster] Waiting for workers to become healthy...")
    for url in worker_urls:
        if wait_for_health(url):
            print(f"  ✔ {url} — healthy")
        else:
            print(f"  ✗ {url} — FAILED to start within timeout")

    # ── Launch router ────────────────────────────────────────────────────────
    print(f"\n[Cluster] Starting router on port {router_port}...")
    worker_args = []
    for url in worker_urls:
        worker_args += ["--workers", url]

    router_proc = subprocess.Popen(
        [sys.executable, "-m", "distributed.router",
         "--port", str(router_port)] + worker_args
    )
    procs.append(router_proc)

    if wait_for_health(f"http://localhost:{router_port}"):
        print(f"  ✔ Router — healthy at http://localhost:{router_port}")
    else:
        print(f"  ✗ Router failed to start")

    # ── Cluster summary ──────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  Cluster is UP!")
    print(f"{'='*55}")
    print(f"  Router (send requests here): http://localhost:{router_port}/generate")
    print(f"  Worker stats:                http://localhost:{router_port}/stats")
    for url in worker_urls:
        print(f"  Worker health:               {url}/health")
    print(f"\n  Press Ctrl+C to shut down the cluster.\n")

    try:
        # Keep alive until Ctrl+C
        for proc in procs:
            proc.wait()
    except KeyboardInterrupt:
        print("\n[Cluster] Shutting down...")
        for proc in procs:
            proc.terminate()
        for proc in procs:
            proc.wait(timeout=5)
        print("[Cluster] Stopped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch distributed ML inference cluster")
    parser.add_argument("--workers", type=int, default=2, help="Number of worker nodes")
    parser.add_argument("--base-port", type=int, default=8001, help="Starting port for workers")
    parser.add_argument("--router-port", type=int, default=8080, help="Router port")
    args = parser.parse_args()

    launch_cluster(args.workers, args.base_port, args.router_port)
