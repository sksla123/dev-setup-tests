import threading
import multiprocessing as mp
import time
import os
import sys

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
    # 파이썬 버전 및 GIL 상태 확인
    print(f"Python Version: {sys.version}")
    
    # Python 3.13+의 free-threaded 빌드 여부 확인
    # sys._is_gil_enabled()는 3.13 free-threaded 빌드에서만 False를 반환할 수 있음
    # 그 외 일반적인 파이썬 환경에서는 속성이 없거나 True를 반환한다고 가정
    if hasattr(sys, "_is_gil_enabled"):
        gil_enabled = sys._is_gil_enabled()
    else:
        gil_enabled = True
        
    status_msg = "Enabled (Active)" if gil_enabled else "Disabled (Free-threading)"
    print(f"GIL Status: {status_msg}")
    print("-" * 40)

    cores = os.cpu_count()
    print(f"CPU cores: {cores}")

    t_thread = run_threads(cores)
    print(f"threading ({cores} threads): {t_thread:.2f}s")

    t_proc = run_processes(cores)
    print(f"multiprocessing ({cores} processes): {t_proc:.2f}s")