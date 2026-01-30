import time
import sys
import os
import subprocess
from numba import njit

# ----------------------------------------------------------------
# 설정
# ----------------------------------------------------------------
N = 500_000_000

# ----------------------------------------------------------------
# 연산 함수 정의
# ----------------------------------------------------------------
def compute_python_loop():
    s = 0
    for i in range(N):
        s += i * i
    return s

@njit
def compute_numba():
    s = 0
    for i in range(N):
        s += i * i
    return s

# ----------------------------------------------------------------
# 워커(Worker): 실제 벤치마크 수행
# ----------------------------------------------------------------
def run_benchmark_worker():
    # 현재 JIT 활성화 상태 확인
    is_jit = getattr(sys, "_is_jit_enabled", lambda: False)()
    mode_name = "Python (CPython JIT Enabled)" if is_jit else "Pure Python (Interpreted)"
    
    print(f"[{mode_name}] 시작 (PID: {os.getpid()})...")
    
    # 1. Python Loop
    start = time.perf_counter()
    compute_python_loop()
    elapsed_python = time.perf_counter() - start
    
    # 2. Numba (Warm-up 포함)
    compute_numba() 
    start_numba = time.perf_counter()
    compute_numba()
    elapsed_numba = time.perf_counter() - start_numba
    
    # 결과 출력 (파싱하기 쉽게 포맷팅)
    print(f"RESULT|{mode_name}|{elapsed_python:.4f}|{elapsed_numba:.6f}")

# ----------------------------------------------------------------
# 오케스트레이터(Orchestrator): 하위 프로세스 관리
# ----------------------------------------------------------------
def run_orchestrator():
    print("=" * 70)
    print(f"Benchmarks Orchestrator (Python {sys.version.split()[0]})")
    print("=" * 70)

    results = []

    # 시나리오 정의: (표시 이름, 실행 옵션 리스트)
    scenarios = [
        ("Interpreted", []),                # 옵션 없음 (JIT Off)
        ("JIT Enabled", ["-X", "jit"])      # JIT On
    ]

    current_script = os.path.abspath(__file__)

    for label, flags in scenarios:
        cmd = [sys.executable] + flags + [current_script, "--worker"]
        
        print(f"Running: {label} Mode...", end=" ", flush=True)
        try:
            # subprocess로 자기 자신 호출
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            # 자식 프로세스 출력 파싱
            for line in result.stdout.splitlines():
                if line.startswith("RESULT|"):
                    parts = line.split("|")
                    mode = parts[1]
                    t_py = float(parts[2])
                    t_nb = float(parts[3])
                    results.append((mode, t_py, t_nb))
            print("Done.")
            
        except subprocess.CalledProcessError as e:
            print(f"Error!\n{e.stderr}")

    # 최종 종합 리포트 출력
    print("\n" + "=" * 70)
    print(f"{'Mode':<30} | {'Pure Python (s)':<15} | {'Numba (s)':<15} | {'Speedup'}")
    print("-" * 70)
    
    base_time = None

    for mode, t_py, t_nb in results:
        speedup_str = f"{t_py / t_nb:,.1f}x" if t_nb > 0 else "N/A"
        print(f"{mode:<30} | {t_py:<15.4f} | {t_nb:<15.6f} | {speedup_str}")
        
        if "Interpreted" in mode:
            base_time = t_py
        elif base_time:
            jit_improvement = (base_time - t_py) / base_time * 100
            print(f"   L-> CPython JIT Improvement over Interpreted: {jit_improvement:.2f}%")

    print("=" * 70)

# ----------------------------------------------------------------
# 진입점
# ----------------------------------------------------------------
if __name__ == "__main__":
    # 인자에 --worker가 있으면 일꾼 모드, 없으면 지휘자 모드
    if "--worker" in sys.argv:
        run_benchmark_worker()
    else:
        run_orchestrator()