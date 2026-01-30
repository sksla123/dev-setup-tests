import tensorflow as tf
import sys
import platform
import os
import argparse

# TensorFlow 로그 레벨 조정 (불필요한 정보 숨기기)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def print_header(title):
    print("\n" + "="*60)
    print(f" {title} ")
    print("="*60)

def get_env_info():
    print_header("시스템 및 환경 정보")
    print(f"운영체제       : {platform.system()} {platform.release()}")
    print(f"Python 버전    : {sys.version.split()[0]}")
    print(f"TensorFlow 버전: {tf.__version__}")

    try:
        build_info = tf.sysconfig.get_build_info()
        if 'cuda_version' in build_info:
            print(f"빌드된 CUDA 버전 : {build_info['cuda_version']}")
        if 'cudnn_version' in build_info:
            print(f"빌드된 cuDNN 버전: {build_info['cudnn_version']}")
        if 'is_cuda_build' in build_info:
            print(f"CUDA 빌드 여부   : {build_info['is_cuda_build']}")
    except Exception:
        print("빌드 세부 정보를 가져올 수 없습니다.")

def test_tensor_operation(device_name):
    print(f"\n[테스트 대상 장치: {device_name}]")
    try:
        with tf.device(device_name):
            size = 2000
            print(f"  - 텐서 생성 및 이동...", end="")
            x = tf.random.uniform((size, size))
            y = tf.random.uniform((size, size))
            print(" OK")

            print(f"  - 행렬 곱셈 연산 (Matrix Multiplication)...", end="")
            z = tf.matmul(x, y)
            _ = z.numpy() 
            print(" PASS")
            return True
    except Exception as e:
        print(" FAILED")
        print(f"  [!] 에러 발생: {e}")
        return False

def run_diagnostics(force_cpu=False):
    get_env_info()

    print_header("물리적 장치(Physical Devices) 감지")
    
    physical_devices = tf.config.list_physical_devices()
    gpus = tf.config.list_physical_devices('GPU')
    cpus = tf.config.list_physical_devices('CPU')
    
    is_mac_metal = (platform.system() == 'Darwin' and len(gpus) > 0)
    
    print(f"전체 감지된 장치 수: {len(physical_devices)}")
    print(f"  - CPU: {len(cpus)}")
    print(f"  - GPU: {len(gpus)} {'(Metal/MPS 포함)' if is_mac_metal else ''}")

    # 1. GPU 테스트 수행 여부 결정
    if gpus:
        print_header("GPU 가속기 테스트")
        for i, device in enumerate(gpus):
            target_device = f"/device:GPU:{i}"
            print(f"장치 ID: {device.name}")
            try:
                details = tf.config.experimental.get_device_details(device)
                if details:
                    print(f"  - 상세 정보: {details}")
            except:
                pass
            test_tensor_operation(target_device)
    else:
        print("\n[!] 감지된 GPU가 없습니다.")

    # 2. CPU 테스트 수행 여부 결정 (GPU가 없거나 --cpu 플래그가 있을 때)
    if not gpus or force_cpu:
        title = "CPU 테스트 (GPU 미감지)" if not gpus else "CPU 테스트 (명시적 요청)"
        print_header(title)
        test_tensor_operation("/device:CPU:0")
    else:
        print("\n" + "="*60)
        print(" GPU가 정상 감지되어 CPU 테스트를 건너뜁니다.")
        print(" (CPU 테스트를 원하시면 --cpu 인자를 사용하세요.)")
        print("="*60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TensorFlow Installation and Device Diagnostic Tool")
    parser.add_argument(
        '--cpu', 
        action='store_true', 
        help='GPU가 감지되더라도 CPU 테스트를 강제로 수행합니다.'
    )
    
    args = parser.parse_args()
    run_diagnostics(force_cpu=args.cpu)