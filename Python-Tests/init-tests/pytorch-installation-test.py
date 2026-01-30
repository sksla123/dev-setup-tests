import torch
import sys
import platform
import argparse

def print_header(title):
    print("\n" + "="*60)
    print(f" {title} ")
    print("="*60)

def get_env_info():
    print_header("시스템 및 환경 정보")
    print(f"운영체제       : {platform.system()} {platform.release()}")
    print(f"Python 버전    : {sys.version.split()[0]}")
    print(f"PyTorch 버전   : {torch.__version__}")
    
    # 가속기 백엔드 라이브러리 정보
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
        size = 2000 
        print(f"  - 텐서 생성 및 이동 ({device_str})...", end="")
        x = torch.rand(size, size).to(device)
        y = torch.rand(size, size).to(device)
        print(" OK")

        print(f"  - 행렬 곱셈 연산 (Matrix Multiplication)...", end="")
        torch.matmul(x, y)
        
        # 각 디바이스별 동기화
        if 'cuda' in device_str:
            torch.cuda.synchronize()
        elif device_str == 'mps':
            torch.mps.synchronize()
        elif device_str == 'xpu':
            torch.xpu.synchronize()
            
        print(" PASS")
        return True
    except Exception as e:
        print(" FAILED")
        print(f"  [!] 에러 발생: {e}")
        return False

def check_cuda():
    if not torch.cuda.is_available():
        return False
    print_header("NVIDIA CUDA / AMD ROCm 감지됨")
    count = torch.cuda.device_count()
    for i in range(count):
        props = torch.cuda.get_device_properties(i)
        print(f"\n[Device {i}: {props.name}]")
        print(f"  - VRAM Total    : {props.total_memory / 1024**2:.0f} MB")
        print(f"  - SM 개수       : {props.multi_processor_count}")
        print(f"  - Compute Cap   : {props.major}.{props.minor}")
        test_tensor_operation(f"cuda:{i}")
    return True

def check_mps():
    if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
        return False
    print_header("Apple MPS (Metal Performance Shaders) 감지됨")
    print("장치 정보: Apple Silicon GPU (M-series)")
    test_tensor_operation("mps")
    return True

def check_xpu():
    if not hasattr(torch, "xpu") or not torch.xpu.is_available():
        return False
    print_header("Intel XPU 감지됨")
    count = torch.xpu.device_count()
    for i in range(count):
        props = torch.xpu.get_device_properties(i)
        print(f"\n[Device {i}: {props.name}]")
        print(f"  - VRAM Total    : {props.total_memory / 1024**2:.0f} MB")
        test_tensor_operation(f"xpu:{i}")
    return True

def run_full_diagnostic(force_cpu=False):
    get_env_info()
    
    accelerators = [
        ("CUDA/ROCm", check_cuda),
        ("MPS", check_mps),
        ("XPU", check_xpu)
    ]
    
    found_any = False
    for name, check_func in accelerators:
        if check_func():
            found_any = True
    
    # GPU 가속기가 하나도 없거나, 사용자가 --cpu 플래그를 넣었을 때만 CPU 테스트 수행
    if not found_any or force_cpu:
        header_msg = "CPU 테스트 (가속기 미감지)" if not found_any else "CPU 테스트 (명시적 요청)"
        print_header(header_msg)
        print(f"프로세서 정보: {platform.processor()}")
        test_tensor_operation("cpu")
        print("\n" + "="*60)
    else:
        print("\n" + "="*60)
        print(" 가속기(GPU/MPS/XPU) 테스트가 완료되었습니다.")
        print(" CPU 테스트를 건너뜁니다. (원할 경우 --cpu 인자 사용)")
        print("="*60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Installation and Accelerator Diagnostic Tool")
    parser.add_argument(
        '--cpu', 
        action='store_true', 
        help='가속기가 감지되더라도 CPU 테스트를 강제로 수행합니다.'
    )
    
    args = parser.parse_args()
    run_full_diagnostic(force_cpu=args.cpu)