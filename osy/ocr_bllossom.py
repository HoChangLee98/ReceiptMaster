import os
from paddleocr import PaddleOCR
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import time

# PaddleOCR 초기화 (GPU 비활성화 옵션으로 경량화 가능)
ocr = PaddleOCR(use_angle_cls=True, lang='korean', use_gpu=False)  

# 이미지 경로 설정
image_path = "./test.jpeg"  # 처리할 이미지 경로

# 1. PaddleOCR로 텍스트 추출
ocr_result = ocr.ocr(image_path, cls=True)
extracted_texts = [line[1][0] for line in ocr_result[0]]

# 추출된 텍스트 출력
print("Extracted Texts from Image:")
print(extracted_texts)

# 2. 모델 및 토크나이저 초기화
print("Initializing model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("Bllossom/llama-3.2-Korean-Bllossom-3B", legacy=False)

# Padding token 설정
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # eos_token을 pad_token으로 사용

model = AutoModelForCausalLM.from_pretrained("Bllossom/llama-3.2-Korean-Bllossom-3B")

# 디바이스 설정 (CPU로 경량화)
device = torch.device("cpu")
print(f"Using device: {device}")
model = model.to(device)

# 3. 추론 함수
def process_with_llama(input_texts, model, tokenizer, max_length=50, timeout=10):
    results = []
    for i, text in enumerate(input_texts):
        print(f"Processing text {i+1}/{len(input_texts)}: {text[:50]}...")
        start_time = time.time()
        try:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).to(device)
            outputs = model.generate(
                inputs["input_ids"], 
                max_length=max_length, 
                pad_token_id=tokenizer.pad_token_id
            )
            elapsed_time = time.time() - start_time
            if elapsed_time > timeout:
                print(f"Warning: Text {i+1} processing took too long ({elapsed_time:.2f} seconds).")
            results.append(tokenizer.decode(outputs[0], skip_special_tokens=True))
        except Exception as e:
            print(f"Error processing text {i+1}: {e}")
            results.append("Error during inference.")
        finally:
            elapsed_time = time.time() - start_time
            print(f"Finished text {i+1}/{len(input_texts)} in {elapsed_time:.2f} seconds.")
    return results

# 첫 번째 텍스트만 처리
if extracted_texts:
    print("Starting inference...")
    processed_texts = process_with_llama(extracted_texts[:1], model, tokenizer, max_length=50)
else:
    processed_texts = ["No text detected."]

# 결과 출력
print("\nProcessed Texts with Llama:")
for original, processed in zip(extracted_texts, processed_texts):
    print(f"Original: {original}")
    print(f"Processed: {processed}")
    print("-" * 30)
