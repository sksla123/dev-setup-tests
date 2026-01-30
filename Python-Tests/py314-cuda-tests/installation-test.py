import torch
import sys
import platform

def run_test():
    print("="*50)
    print(" [환경 정보 확인] ")
    print("="*50)
    
    # OS 및 파이썬 정보
    print(f"운영체제: {platform.system()} {platform.release()}")
    print(f"파이썬 버전: {sys.version}")
    
    # PyTorch 정보
    print(f"PyTorch 버전: {torch.__version__}")
    
    # CUDA 및 cuDNN 정보
    cuda_available = torch.cuda.is_available()
    print(f"CUDA 사용 가능 여부: {'SUCCESS' if cuda_available else 'FAILED'}")
    
    if cuda_available:
        print(f"PyTorch가 인식한 CUDA 버전: {torch.version.cuda}")
        print(f"PyTorch가 인식한 cuDNN 버전: {torch.backends.cudnn.version()}")
        
        # GPU 하드웨어 정보
        current_device = torch.cuda.current_device()
        gpu_properties = torch.cuda.get_device_properties(current_device)
        
        print("\n" + "="*50)
        print(" [GPU 하드웨어 정보] ")
        print("="*50)
        print(f"모델명: {torch.cuda.get_device_name(0)}")
        print(f"VRAM 총량: {gpu_properties.total_memory / 1024**2:.0f} MB")
        print(f"멀티프로세서(SM) 개수: {gpu_properties.multi_processor_count}")
        print(f"연산 능력(Compute Capability): {gpu_properties.major}.{gpu_properties.minor}")
        
        # 실제 연산 테스트
        print("\n" + "="*50)
        print(" [간이 연산 테스트] ")
        print("="*50)
        try:
            x = torch.rand(1000, 1000).cuda()
            y = torch.rand(1000, 1000).cuda()
            z = torch.matmul(x, y)
            print("GPU 텐서 행렬 곱셈 테스트: PASS")
        except Exception as e:
            print(f"GPU 연산 테스트 실패: {e}")
    else:
        print("\n[!] 경고: GPU를 찾을 수 없습니다. 설치된 패키지가 CPU 전용인지 확인하세요.")

    print("="*50)

if __name__ == "__main__":
    run_test()
