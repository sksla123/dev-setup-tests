# jit-test.py (JIT 없음)
import time

N = 50_000_000

def compute():
    s = 0
    for i in range(N):
        s += i * i
    return s

start = time.time()
compute()
elapsed = time.time() - start

print(f"no JIT elapsed: {elapsed:.2f}s")
