import cv2
import numpy as np

# 1. 버전 확인
print(f"OpenCV 버전: {cv2.__version__}")

# 2. 간단한 검은색 이미지 생성 (이미지 파일이 없을 경우 대비)
img = np.zeros((400, 400, 3), dtype=np.uint8)
cv2.putText(img, "OpenCV Test Success!", (50, 200), 
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

# 3. 이미지 창 띄우기
cv2.imshow("OpenCV Test", img)

print("아무 키나 누르면 창이 닫힙니다.")
cv2.waitKey(0)
cv2.destroyAllWindows()