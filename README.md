# ReceiptMaster 
![pig_img](https://github.com/user-attachments/assets/0adceb8e-7c80-4ded-8ee6-99a536fd9898)

**ReceiptMaster**는 영수증 이미지를 입력으로 받아 **텍스트를 추출(PaddleOCR)**하고, 추출된 텍스트를 **구조화된 데이터(LLaMA)**로 변환하여 **CSV 파일 또는 exel 파일로 저장**하는 AI 기반 데이터 처리 파이프라인입니다.  
이 프로젝트는 **영수증 데이터의 디지털화와 자동화를 목표**로 합니다.

---

## Features

- **영수증 이미지 → CSV/EXCEL 변환**:
  - PaddleOCR로 영수증에서 텍스트를 추출.
  - OPENAI를 사용하여 비정형 텍스트를 JSON 형식으로 구조화.
  - 최종 결과를 CSV 파일 또는 EXCEL 파일 등 원하는 파일로 저장.

- **다국어 지원**:
  - 영수증 텍스트에 대한 다국어 인식 가능(영어, 한글 등)하나 현재 프로젝트는 한국어에 집중

- **유연한 데이터 구조화**:
  - 사용자 정의 JSON 형식으로 데이터 출력.

- **빠르고 정확한 처리**:
  - PaddleOCR 경량화 모델과 OPENAI를 최적화하여 빠르고 정확한 결과 제공.

---

## Installation

### Requirements

- Python 3.8 이상
- 필수 라이브러리:
  - `paddleocr`
  - `transformers`
  - `pandas`
  - `opencv-python`

### Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/receiptmaster.git
   cd receiptmaster
