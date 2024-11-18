import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import cv2
import time
from matplotlib import pyplot as plt
from datetime import datetime

# Ko-TroCR 모델 및 프로세서 로드
processor = TrOCRProcessor.from_pretrained("ddobokki/ko-trocr")
model = VisionEncoderDecoderModel.from_pretrained("ddobokki/ko-trocr")

# GPU 또는 CPU 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

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
beta = 5     # 밝기 값 조정
image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
contrast_time = time.time() - start_time

# 노이즈 제거
start_time = time.time()
image = cv2.fastNlMeansDenoisingColored(image, None, h=3, hColor=3, templateWindowSize=7, searchWindowSize=21)
denoise_time = time.time() - start_time

# 2. Ko-TroCR 추론
start_time = time.time()

# Ko-TroCR 입력 형태로 변환
image_pil = Image.fromarray(image)
pixel_values = processor(images=image_pil, return_tensors="pt").pixel_values.to(device)

# 텍스트 추론
with torch.no_grad():
    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

ocr_time = time.time() - start_time

# 결과 출력
print("Ko-TroCR 추출된 텍스트:")
print(generated_text.encode("utf-8", "replace").decode("utf-8", "replace"))

# 현재 날짜와 시간 가져오고 시각화된 결과를 파일로 저장
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")  # YYYYMMDD_HHMMSS 형식
output_path = f'./ocr_result_{current_time}.jpg'

# Ko-TroCR 결과 이미지에 텍스트 표시
output_image = image.copy()
font_scale = 1.5
font_thickness = 2
color = (0, 255, 0)

# OpenCV로 텍스트 추가
cv2.putText(output_image, generated_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, font_thickness)

cv2.imwrite(output_path, cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR))
plt.figure(figsize=(10, 10))
plt.imshow(output_image)
plt.axis('off')
plt.title("Ko-TroCR Result")
plt.show()

# 소요 시간 출력
print("\n### 시간 측정 결과 ###")
print(f"이미지 로드 시간: {load_time:.4f}초")
print(f"해상도 확대 시간: {resize_time:.4f}초")
print(f"밝기 및 대비 조정 시간: {contrast_time:.4f}초")
print(f"노이즈 제거 시간: {denoise_time:.4f}초")
print(f"Ko-TroCR 추론 시간: {ocr_time:.4f}초")
