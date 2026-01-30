import tensorflow as tf
import sys
import platform
import os

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

    # TensorFlow 빌드 정보 확인 (CUDA/cuDNN 버전 등)
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
    """
    지정된 디바이스(device_name) 컨텍스트에서 행렬 곱셈을 수행합니다.
    """
    print(f"\n[테스트 대상 장치: {device_name}]")
    
    try:
        # 해당 장치로 강제 할당
        with tf.device(device_name):
            size = 2000
            print(f"  - 텐서 생성 및 이동...", end="")
            # 랜덤 텐서 생성
            x = tf.random.uniform((size, size))
            y = tf.random.uniform((size, size))
            print(" OK")

            print(f"  - 행렬 곱셈 연산 (Matrix Multiplication)...", end="")
            # 연산 수행
            z = tf.matmul(x, y)
            
            # 연산 강제 실행 확인 (Eager execution이므로 즉시 실행되나 확실히 하기 위해)
            _ = z.numpy() 
            print(" PASS")
            return True
            
    except RuntimeError as e:
        print(" FAILED")
        print(f"  [!] 런타임 에러: {e}")
    except Exception as e:
        print(" FAILED")
        print(f"  [!] 에러 발생: {e}")
    
    return False

def run_diagnostics():
    get_env_info()

    print_header("물리적 장치(Physical Devices) 감지")
    
    # 모든 물리 장치 조회
    physical_devices = tf.config.list_physical_devices()
    
    gpus = tf.config.list_physical_devices('GPU')
    cpus = tf.config.list_physical_devices('CPU')
    
    # Apple Silicon (Metal) 확인: TF에서는 보통 'GPU'로 잡힙니다.
    # 명시적으로 플러그인이 있는지 확인
    is_mac_metal = (platform.system() == 'Darwin' and len(gpus) > 0)
    
    print(f"전체 감지된 장치 수: {len(physical_devices)}")
    print(f"  - CPU: {len(cpus)}")
    print(f"  - GPU: {len(gpus)} {'(Metal/MPS 포함)' if is_mac_metal else ''}")

    # ---------------------------------------------------------
    # GPU 테스트
    # ---------------------------------------------------------
    if gpus:
        print_header("GPU 가속기 테스트")
        for i, device in enumerate(gpus):
            # device.name 예: '/physical_device:GPU:0'
            # 사용 시에는 '/device:GPU:0' 형태로 매핑됨
            target_device = f"/device:GPU:{i}"
            
            print(f"장치 ID: {device.name}")
            # TF는 VRAM 정보를 파이썬 API로 직접 조회하기 어렵습니다.
            # 대신 메모리 증가 옵션 등을 출력할 수는 있습니다.
            try:
                details = tf.config.experimental.get_device_details(device)
                if details:
                    print(f"  - 상세 정보: {details}")
            except:
                pass
            
            test_tensor_operation(target_device)
    else:
        print("\n[!] GPU를 찾을 수 없습니다.")

    # ---------------------------------------------------------
    # CPU 테스트 (GPU가 없거나 비교용)
    # ---------------------------------------------------------
    if not gpus:
        print_header("CPU (Fallback) 테스트")
        test_tensor_operation("/device:CPU:0")
    else:
        print("\n" + "="*60)
        print(" [참고] GPU 테스트가 완료되었습니다.")
        print(" CPU 테스트는 생략합니다.")
        print("="*60)

if __name__ == "__main__":
    run_diagnostics()