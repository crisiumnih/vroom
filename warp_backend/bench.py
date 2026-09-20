"""Throughput: 100k electrical-RHS evals, numpy vs warp, batch scaling."""
import time
import numpy as np
import warp as wp
from warp_backend import ref
from warp_backend.test_parity import run_warp

wp.init()
rng = np.random.default_rng(3)
N_EVALS = 100_000


def bench_numpy(n):
    states = np.zeros((n, 4), dtype=np.float32)
    states[:, 0:3] = rng.uniform(-4, 4, (n, 3)).astype(np.float32)
    states[:, 3] = rng.uniform(-10, 10, n).astype(np.float32)
    u = rng.uniform(-22, 22, (n, 3)).astype(np.float32)
    w = rng.uniform(-300, 300, n).astype(np.float32)
    reps = max(1, N_EVALS // n)
    t = time.perf_counter()
    for _ in range(reps):
        if n == 1:
            ref.electrical_rhs(states[0], u[0], float(w[0]))
        else:
            ref.electrical_rhs(states, u, w)
    return (time.perf_counter() - t) / (reps * n) * 1e6


def bench_warp(n):
    states = np.zeros((n, 4), dtype=np.float32)
    states[:, 0:3] = rng.uniform(-4, 4, (n, 3)).astype(np.float32)
    states[:, 3] = rng.uniform(-10, 10, n).astype(np.float32)
    u = rng.uniform(-22, 22, (n, 3)).astype(np.float32)
    w = rng.uniform(-300, 300, n).astype(np.float32)
    run_warp(states, u, w)  # warmup / compile
    reps = max(1, N_EVALS // n)
    t = time.perf_counter()
    for _ in range(reps):
        run_warp(states, u, w)
    return (time.perf_counter() - t) / (reps * n) * 1e6


if __name__ == "__main__":
    print(f"numpy  N=1    : {bench_numpy(1):8.2f} us/eval")
    print(f"numpy  N=1024 : {bench_numpy(1024):8.2f} us/eval")
    print(f"warp   N=1    : {bench_warp(1):8.2f} us/eval")
    print(f"warp   N=1024 : {bench_warp(1024):8.2f} us/eval")
    print(f"warp   N=16384: {bench_warp(16384):8.2f} us/eval")
