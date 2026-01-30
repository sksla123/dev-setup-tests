import torch
import sys
import platform

def print_header(title):
    print("\n" + "="*60)
    print(f" {title} ")
    print("="*60)

def get_env_info():
    print_header("시스템 및 환경 정보")
    print(f"운영체제       : {platform.system()} {platform.release()}")
    print(f"Python 버전    : {sys.version.split()[0]}")
    print(f"PyTorch 버전   : {torch.__version__}")
    
    # AMD ROCm 확인 (CUDA 인터페이스를 공유하므로 별도 확인 필요)
    if hasattr(torch.version, 'hip') and torch.version.hip:
        print(f"ROCm (HIP) 버전: {torch.version.hip}")
    elif torch.version.cuda:
        print(f"CUDA 버전      : {torch.version.cuda}")
        if hasattr(torch.backends, 'cudnn'):
            print(f"cuDNN 버전     : {torch.backends.cudnn.version()}")

def test_tensor_operation(device_str):
    """
    지정된 디바이스로 텐서를 이동시키고 연산을 수행하여 테스트합니다.
    """
    try:
        device = torch.device(device_str)
        # 테스트용 데이터 생성 (크기 조절 가능)
        size = 2000 
        print(f"  - 텐서 생성 및 이동 ({device_str})...", end="")
        x = torch.rand(size, size).to(device)
        y = torch.rand(size, size).to(device)
        print(" OK")

        print(f"  - 행렬 곱셈 연산 (Matrix Multiplication)...", end="")
        torch.matmul(x, y) # 동기화가 필요할 수 있으나 기본 테스트로는 충분
        
        # GPU/MPS 등의 비동기 연산 대기를 위해 동기화 수행
        if device_str == 'cuda':
            torch.cuda.synchronize()
        elif device_str == 'mps':
            torch.mps.synchronize()
        elif device_str == 'xpu':
            torch.xpu.synchronize()
            
        print(" PASS")
        return True
    except Exception as e:
        print(f" FAILED")
        print(f"  [!] 에러 발생: {e}")
        return False

def check_cuda():
    if not torch.cuda.is_available():
        return False
    
    print_header("NVIDIA CUDA / AMD ROCm 감지됨")
    count = torch.cuda.device_count()
    print(f"감지된 장치 수: {count}")

    for i in range(count):
        props = torch.cuda.get_device_properties(i)
        print(f"\n[Device {i}: {props.name}]")
        print(f"  - VRAM Total    : {props.total_memory / 1024**2:.0f} MB")
        print(f"  - SM 개수       : {props.multi_processor_count}")
        print(f"  - Compute Cap   : {props.major}.{props.minor}")
        
        test_tensor_operation(f"cuda:{i}")
    return True

def check_mps():
    # macOS Apple Silicon (M1/M2/M3 등)
    if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
        return False
        
    print_header("Apple MPS (Metal Performance Shaders) 감지됨")
    print("장치 정보: Apple Silicon GPU (M-series)")
    # MPS는 구체적인 하드웨어 스펙 쿼리 API가 제한적이므로 연산 테스트 위주로 수행
    test_tensor_operation("mps")
    return True

def check_xpu():
    # Intel GPU (Arc 등) - 최신 파이토치 확장 필요
    if not hasattr(torch, "xpu") or not torch.xpu.is_available():
        return False

    print_header("Intel XPU 감지됨")
    count = torch.xpu.device_count()
    print(f"감지된 장치 수: {count}")
    
    for i in range(count):
        props = torch.xpu.get_device_properties(i)
        print(f"\n[Device {i}: {props.name}]")
        print(f"  - VRAM Total    : {props.total_memory / 1024**2:.0f} MB")
        test_tensor_operation(f"xpu:{i}")
    return True

def check_cpu():
    print_header("CPU (Fallback) 테스트")
    # CPU 정보는 platform 모듈이나 os 모듈로 일부 확인 가능하나, 여기서는 연산 위주
    print(f"프로세서 정보: {platform.processor()}")
    test_tensor_operation("cpu")

def run_full_diagnostic():
    get_env_info()
    
    found_accelerator = False
    
    # 1. CUDA (NVIDIA/AMD) 체크
    if check_cuda():
        found_accelerator = True
        
    # 2. MPS (Apple Silicon) 체크
    if check_mps():
        found_accelerator = True
        
    # 3. XPU (Intel) 체크
    if check_xpu():
        found_accelerator = True
        
    # 가속기를 찾지 못했거나, 명시적으로 CPU 비교를 원할 경우를 대비해 마지막엔 CPU 상태도 확인
    if not found_accelerator:
        print("\n[!] GPU 가속기를 찾을 수 없습니다. CPU 테스트를 수행합니다.")
        check_cpu()
    else:
        print("\n" + "="*60)
        print(" 가속기(GPU/MPS) 테스트가 완료되었습니다.")
        print(" CPU 테스트는 생략합니다.")
        print("="*60)

if __name__ == "__main__":
    run_full_diagnostic()