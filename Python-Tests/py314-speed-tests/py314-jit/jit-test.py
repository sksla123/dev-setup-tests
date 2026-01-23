# jit.py (JIT 있음)
import time
from numba import njit

N = 50_000_000

@njit
def compute():
    s = 0
    for i in range(N):
        s += i * i
    return s

# warm-up (JIT compile)
compute()

start = time.time()
compute()
elapsed = time.time() - start

print(f"JIT elapsed: {elapsed:.2f}s")
