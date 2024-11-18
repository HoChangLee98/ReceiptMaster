import cv2
from paddleocr import PaddleOCR, draw_ocr
from matplotlib import pyplot as plt
import os
from datetime import datetime
import time

from paddleocr import __version__
print(f"PaddleOCR Version: {__version__}")


# PaddleOCR 초기화
ocr = PaddleOCR(
    # use_angle_cls=True,
    use_angle_cls=False,
    lang='korean',  # 사용할 언어 설정
    use_gpu=False,  # GPU 사용 여부
    det_model_dir="./ch_PP-OCRv4_det_infer",  # 텍스트 감지 모델 경로
    rec_model_dir="./korean_PP-OCRv3_rec_infer"  # 텍스트 인식 모델 경로
)


# 영수증 이미지 경로
image_path = './test.jpeg'  # 영수증 이미지 파일 경로

# 1. 이미지 로드
start_time = time.time()
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # OpenCV는 BGR로 로드하므로 RGB로 변환
load_time = time.time() - start_time


# 해상도 확대
start_time = time.time()
image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
resize_time = time.time() - start_time


# 밝기 및 대비 조정
start_time = time.time()
alpha = 1.1  # 대비 값 조정
beta = 5    # 밝기 값 조정
image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
contrast_time = time.time() - start_time


# 노이즈 제거
start_time = time.time()
image = cv2.fastNlMeansDenoisingColored(image, 
                                        None, 
                                        h=3, 
                                        hColor=3, 
                                        templateWindowSize=7, 
                                        searchWindowSize=21)
denoise_time = time.time() - start_time

# 2. OCR 추론
start_time = time.time()
ocr_result = ocr.ocr(image, cls=True)
ocr_time = time.time() - start_time

# 신뢰도 기반 필터링
filtered_results = [line for line in ocr_result[0] if line[1][1] > 0.6]

# 필터링된 텍스트에서 특정 키워드 추출
# desired_keywords = ["상호", "신한카드", "총합계", "누적P"]
# extracted_texts = {}

# for line in filtered_results:
#     text = line[1][0]
#     for keyword in desired_keywords:
#         if keyword in text:
#             extracted_texts[keyword] = text

# print("추출된 텍스트 (신뢰도 기반):")
# for key, value in extracted_texts.items():
#     print(f"{key}: {value}")

# 결과 시각화
boxes = [line[0] for line in filtered_results]
texts = [line[1][0] for line in filtered_results]
scores = [line[1][1] for line in filtered_results]

image_with_boxes = draw_ocr(image, boxes, texts, scores, font_path='./fonts/NanumSquareNeo-bRg.ttf')
show_img = draw_ocr(image, boxes)  # 폰트 경로 설정


# 현재 날짜와 시간 가져오고 시각화된 결과를 파일로 저장
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")  # YYYYMMDD_HHMMSS 형식
output_path = f'./ocr_result_{current_time}.jpg'
cv2.imwrite(output_path, cv2.cvtColor(image_with_boxes, cv2.COLOR_RGB2BGR))  # RGB -> BGR로 변환 후 저장
plt.figure(figsize=(10, 10))
plt.imshow(show_img)
plt.axis('off')
plt.title("OCR Result")
plt.show()


# 소요 시간 출력
print("\n### 시간 측정 결과 ###")
print(f"이미지 로드 시간: {load_time:.4f}초")
print(f"해상도 확대 시간: {resize_time:.4f}초")
print(f"밝기 및 대비 조정 시간: {contrast_time:.4f}초")
print(f"노이즈 제거 시간: {denoise_time:.4f}초")
print(f"OCR 추론 시간: {ocr_time:.4f}초")