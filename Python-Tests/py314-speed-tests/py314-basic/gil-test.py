import threading
import multiprocessing as mp
import time
import os

N = 50_000_000

def cpu_bound():
    x = 0
    for i in range(N):
        x += i * i
    return x

def cpu_bound_wrapper(_):
    return cpu_bound()

def run_threads(n_threads):
    threads = []
    start = time.time()
    for _ in range(n_threads):
        t = threading.Thread(target=cpu_bound)
        t.start()
        threads.append(t)
    for t in threads:
        t.join()
    return time.time() - start

def run_processes(n_procs):
    start = time.time()
    with mp.Pool(n_procs) as p:
        p.map(cpu_bound_wrapper, range(n_procs))
    return time.time() - start

if __name__ == "__main__":
    cores = os.cpu_count()
    print(f"CPU cores: {cores}")

    t_thread = run_threads(cores)
    print(f"threading ({cores} threads): {t_thread:.2f}s")

    t_proc = run_processes(cores)
    print(f"multiprocessing ({cores} processes): {t_proc:.2f}s")
