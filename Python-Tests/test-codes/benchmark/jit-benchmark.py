import time
import sys
import platform
from numba import njit

N = 50_000_000

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

if __name__ == "__main__":
    # ---------------------------------------------------------
    # 1. 환경 정보 확인 및 모드 설정
    # ---------------------------------------------------------
    print("=" * 60)
    print(f"Python Version: {sys.version.split()[0]}")

    interpreter_jit_active = False
    mode_name = "Pure Python (Interpreted)"  # 기본 모드 명칭

    # Python 3.13+ Experimental JIT 확인
    if hasattr(sys, "_is_jit_enabled") and sys._is_jit_enabled():
        interpreter_jit_active = True
        mode_name = "Python (CPython JIT Enabled)"
    # PyPy 확인
    elif platform.python_implementation() == "PyPy":
        interpreter_jit_active = True
        mode_name = "Python (PyPy JIT)"

    print(f"Environment Mode: {mode_name}")
    print("=" * 60)

    # ---------------------------------------------------------
    # 2. Python Loop 벤치마크 (환경에 따라 JIT 또는 인터프리터)
    # ---------------------------------------------------------
    # 동적으로 결정된 이름을 출력에 사용
    print(f"[1] {mode_name} Execution (N={N:,})")
    
    start = time.time()
    compute_python_loop()
    elapsed_python = time.time() - start
    
    print(f"    -> Elapsed: {elapsed_python:.4f}s")
    print("-" * 60)

    # ---------------------------------------------------------
    # 3. Numba (@njit) 벤치마크
    # ---------------------------------------------------------
    print(f"[2] Numba @njit Execution (N={N:,})")
    
    # Warm-up
    compute_numba()
    
    start = time.time()
    compute_numba()
    elapsed_numba = time.time() - start
    
    print(f"    -> Elapsed: {elapsed_numba:.4f}s")
    print("=" * 60)

    # ---------------------------------------------------------
    # 4. 결과 비교
    # ---------------------------------------------------------
    if elapsed_numba > 0:
        speedup = elapsed_python / elapsed_numba
        print(f"Result: Numba is {speedup:.1f}x faster than {mode_name}")