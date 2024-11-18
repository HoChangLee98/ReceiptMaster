import os
import json
from PIL import Image
from tqdm import tqdm

# 경로 설정
BASE_DIR = "./ocr_sample"
SRC_DIR = os.path.join(BASE_DIR, "src_data", "accept")
LABEL_DIR = os.path.join(BASE_DIR, "label_data", "accept")

def collect_file_paths(src_dir, label_dir):
    """
    이미지와 JSON 파일의 경로를 수집합니다.
    """
    image_paths = []
    label_paths = []

    for root, _, files in os.walk(src_dir):
        for file in files:
            if file.endswith(".jpg"):
                image_paths.append(os.path.join(root, file))

    for root, _, files in os.walk(label_dir):
        for file in files:
            if file.endswith(".json"):
                label_paths.append(os.path.join(root, file))

    return sorted(image_paths), sorted(label_paths)

def match_files(image_paths, label_paths):
    """
    이미지 파일과 JSON 파일을 매칭합니다.
    파일명(경로 포함)이 동일한 경우 매칭합니다.
    """
    matched_files = []

    image_dict = {os.path.splitext(os.path.basename(img))[0]: img for img in image_paths}
    label_dict = {os.path.splitext(os.path.basename(lbl))[0]: lbl for lbl in label_paths}

    for key in image_dict.keys():
        if key in label_dict:
            matched_files.append((image_dict[key], label_dict[key]))

    return matched_files

def load_json(json_path):
    """
    JSON 파일을 로드합니다.
    """
    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def display_image_with_labels(image_path, json_data):
    """
    이미지와 JSON 레이블 데이터를 함께 출력합니다.
    """
    image = Image.open(image_path)

    print(f"Image Path: {image_path}")
    print(f"JSON Data: {json.dumps(json_data, indent=4, ensure_ascii=False)}")
    image.show()

def process_data(matched_files):
    """
    매칭된 파일 쌍(이미지와 JSON)을 처리합니다.
    """
    for image_path, json_path in tqdm(matched_files, desc="Processing Files"):
        try:
            # JSON 데이터 로드
            json_data = load_json(json_path)
            
            # 이미지와 JSON 데이터 출력 (필요 시 주석 처리)
            display_image_with_labels(image_path, json_data)
            
        except Exception as e:
            print(f"Error processing {image_path} and {json_path}: {e}")

if __name__ == "__main__":
    # 이미지와 JSON 파일 경로 수집
    image_paths, label_paths = collect_file_paths(SRC_DIR, LABEL_DIR)

    # 이미지와 JSON 파일 매칭
    matched_files = match_files(image_paths, label_paths)

    # 매칭된 데이터 처리
    process_data(matched_files)
