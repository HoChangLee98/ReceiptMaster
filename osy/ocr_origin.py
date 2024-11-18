import cv2
from paddleocr import PaddleOCR, draw_ocr
from matplotlib import pyplot as plt
import os
from datetime import datetime

from paddleocr import __version__
print(f"PaddleOCR Version: {__version__}")


# # PaddleOCR 초기화 (한국어 포함 다국어 모델 사용)
# ocr = PaddleOCR(use_angle_cls=True, 
#                 # lang='en',
#                 lang='korean',
#                 use_gpu=False)  # GPU 사용 시 use_gpu=True

# PaddleOCR 초기화
ocr = PaddleOCR(
    # use_angle_cls=True,
    use_angle_cls=False,
    lang='korean',  # 사용할 언어 설정
    use_gpu=False,  # GPU 사용 여부
    det_model_dir="./ch_PP-OCRv4_det_infer",  # 텍스트 감지 모델 경로
    rec_model_dir="./korean_PP-OCRv3_rec_infer"   # 텍스트 인식 모델 경로
)

# # 이미지 밝기 및 대비 조정 함수
# def adjust_brightness_contrast(image, alpha=1.5, beta=50):
#     """
#     이미지 밝기와 대비를 조정하는 함수.
#     :param image: 입력 이미지 (RGB 포맷)
#     :param alpha: 대비 값 (1.0 이상으로 설정, 기본값: 1.5)
#     :param beta: 밝기 값 (0 이상으로 설정, 기본값: 50)
#     :return: 조정된 이미지
#     """
#     # 밝기와 대비를 적용한 이미지 반환
#     return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)


# 영수증 이미지 경로
image_path = './test.jpeg'  # 영수증 이미지 파일 경로

# 이미지 경로 확인 및 읽기
if not os.path.exists(image_path):
    print(f"Error: File not found at {image_path}")
    exit()

# 1. 이미지 로드
image = cv2.imread(image_path)
if image is None:
    print(f"Error: Unable to read the image file at {image_path}")
    exit()
    
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # OpenCV는 BGR로 로드하므로 RGB로 변환


# 밝기와 대비를 조정한 전처리 이미지 생성
# processed_image = adjust_brightness_contrast(image, alpha=1.3, beta=30)

# 2. OCR 추론 실행
ocr_result = ocr.ocr(image, cls=True)
# ocr_result = ocr.ocr(processed_image, cls=True)

# 3. 텍스트 추출 및 출력
print("Extracted Texts:")
for line in ocr_result[0]:
    text, confidence = line[1]
    print(f"Text: {text}, Confidence: {confidence:.2f}")

# 4. OCR 결과 시각화
# 텍스트와 박스를 이미지에 표시
boxes = [line[0] for line in ocr_result[0]]  # 텍스트 박스 좌표
texts = [line[1][0] for line in ocr_result[0]]  # 텍스트
scores = [line[1][1] for line in ocr_result[0]]  # 신뢰도 점수

# 결과를 시각화 이미지로 그리기
image_with_boxes = draw_ocr(image, boxes, texts, scores, font_path='./fonts/NanumSquareNeo-bRg.ttf')  # 폰트 경로 설정
show_img = draw_ocr(image, boxes)  # 폰트 경로 설정



# 현재 날짜와 시간 가져오기
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")  # YYYYMMDD_HHMMSS 형식

# 시각화된 결과를 파일로 저장
output_path = f'./ocr_result_{current_time}.jpg'
cv2.imwrite(output_path, cv2.cvtColor(image_with_boxes, cv2.COLOR_RGB2BGR))  # RGB -> BGR로 변환 후 저장


plt.figure(figsize=(10, 10))
plt.imshow(show_img)
plt.axis('off')
plt.title("OCR Result")
plt.show()
