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

def require_jit_build_or_exit():
    """
    CPython experimental JIT 빌드가 아니면 경고 출력 후 종료
    """
    if not hasattr(sys, "_is_jit_enabled"):
        print("=" * 70, file=sys.stderr)
        print("ERROR: This Python interpreter is NOT built with CPython JIT support.", file=sys.stderr)
        print("       (sys._is_jit_enabled is missing)", file=sys.stderr)
        print("", file=sys.stderr)
        print("This benchmark is meaningless without a JIT-enabled CPython build.", file=sys.stderr)
        print("Please rebuild CPython with:", file=sys.stderr)
        print("  ./configure --enable-experimental-jit", file=sys.stderr)
        print("=" * 70, file=sys.stderr)
        sys.exit(1)

# ----------------------------------------------------------------
# 워커(Worker): 실제 벤치마크 수행
# ----------------------------------------------------------------
def run_benchmark_worker():
    # 현재 JIT 활성화 상태 확인
    # 3.13+ JIT 여부 확인
    is_jit = False
    if hasattr(sys, "_is_jit_enabled"):
        is_jit = sys._is_jit_enabled()
    
    # 실행 로그는 stderr로 출력해야 캡처되지 않고 화면에 보입니다.
    print(f"Running python path: {sys.executable}", file=sys.stderr)
    
    mode_name = "Python (CPython JIT Enabled)" if is_jit else "Standard Python (Interpreted)"
    
    print(f"[{mode_name}] 시작 (PID: {os.getpid()})...", file=sys.stderr)
    
    # 1. Python Loop
    start = time.perf_counter()
    compute_python_loop()
    elapsed_python = time.perf_counter() - start
    
    # 2. Numba (Warm-up 포함)
    compute_numba() 
    start_numba = time.perf_counter()
    compute_numba()
    elapsed_numba = time.perf_counter() - start_numba
    
    # 결과 데이터는 stdout으로 출력 (Orchestrator가 파싱)
    print(f"RESULT|{mode_name}|{elapsed_python:.4f}|{elapsed_numba:.6f}")

# ----------------------------------------------------------------
# 오케스트레이터(Orchestrator): 하위 프로세스 관리
# ----------------------------------------------------------------
def run_orchestrator():
    require_jit_build_or_exit()

    print("=" * 70)
    print(f"Benchmarks Orchestrator (Python {sys.version.split()[0]})")
    print("=" * 70)

    results = []

    # 시나리오 정의: (표시 이름, 실행 플래그, 환경 변수)
    scenarios = [
        ("Interpreted", [], {}),                                 # JIT Off
        ("JIT Enabled", ["-X", "jit"], {"PYTHON_JIT": "1"})      # JIT On (플래그 + 환경변수)
    ]

    current_script = os.path.abspath(__file__)

    for label, flags, env_vars in scenarios:
        cmd = [sys.executable] + flags + [current_script, "--worker"]
        
        # 부모 환경 변수 복사 후 JIT 설정 추가
        current_env = os.environ.copy()
        current_env.update(env_vars)
        
        print(f"Running: {label} Mode...", end=" ", flush=True)
        try:
            # env 인자를 통해 환경 변수 주입!
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                check=True, 
                env=current_env
            )

            # 캡처된 내용을 화면에도 보여줌 (확인용)
            print("\n", "-" * 70)
            print(f"[{label}] Worker Output (stderr/stdout merged logic handled via pipe)\n")
            # stderr로 보낸 건 이미 화면에 나왔을 것이고, 여기선 stdout(결과)만 찍힘
            print(result.stdout.strip())
            print("-" * 70)
            
            # 자식 프로세스 출력 파싱
            parsed = False
            for line in result.stdout.splitlines():
                if line.startswith("RESULT|"):
                    parts = line.split("|")
                    mode = parts[1]
                    t_py = float(parts[2])
                    t_nb = float(parts[3])
                    results.append((mode, t_py, t_nb))
                    parsed = True
            
            if parsed:
                print("Done.")
            else:
                print("Failed to parse result.")
            
        except subprocess.CalledProcessError as e:
            print(f"Error!\n{e.stderr}")

    # 최종 종합 리포트 출력
    print("\n" + "=" * 70)
    print(f"{'Mode':<30} | {'Standard Python (s)':<15} | {'Numba (s)':<15} | {'Speedup'}")
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
    if "--worker" in sys.argv:
        run_benchmark_worker()
    else:
        run_orchestrator()