
import streamlit as st
import time 

import os
import pandas as pd
from io import BytesIO
import json

from pyxlsb import open_workbook as open_xlsb

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_teddynote.models import MultiModal
from langchain_teddynote.messages import stream_response

from paddleocr import PaddleOCR, draw_ocr
from datetime import datetime

from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt

def to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')
        workbook = writer.book
        worksheet = writer.sheets['Sheet1']
        format1 = workbook.add_format({'num_format': '0.00'}) 
        worksheet.set_column('A:A', None, format1)  
    processed_data = output.getvalue()
    return processed_data

def main():
    st.write('## 캐피털아이')

    ## input 이미지 업로드 
    img_file = st.file_uploader('이미지를 업로드 하세요.', type=['png', 'jpg', 'jpeg'])

    # 파일이 업로드된 경우
    if img_file is not None:
        if 'ocr_result' not in st.session_state:
            
            # OCR 추론 및 결과 저장
            st.write('#### 텍스트 추출 중')
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            # image = cv2.imread(img_file)
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # OpenCV는 BGR로 로드하므로 RGB로 변환

            # 해상도 확대
            image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            
            # 밝기 및 대비 조정
            alpha = 1.1  # 대비 값 조정
            beta = 5    # 밝기 값 조
            image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
            
            # 노이즈 제거
            image = cv2.fastNlMeansDenoisingColored(image, 
                                        None, 
                                        h=3, 
                                        hColor=3, 
                                        templateWindowSize=7, 
                                        searchWindowSize=21)


            with st.spinner('Operating OCR ...'):
                ocr = PaddleOCR(
                    use_angle_cls=False,
                    lang='korean',
                    use_gpu=False,
                    det_model_dir="./ch_PP-OCRv4_det_infer",
                    rec_model_dir="./korean_PP-OCRv3_rec_infer"
                )
                ocr_result = ocr.ocr(image, cls=True)
                text_result = " ".join([line[1][0] for line in ocr_result[0]]).strip()
                
                # 이미지 보여주기
                boxes = [line[0] for line in ocr_result[0]]  # 텍스트 박스 좌표
                texts = [line[1][0] for line in ocr_result[0]]  # 텍스트
                scores = [line[1][1] for line in ocr_result[0]]  # 신뢰도 점수
                font_path = '../nanum-gothic/NanumGothic.ttf'  # 폰트 경로
                
                image_with_boxes = draw_ocr(image, boxes, texts, scores, font_path=font_path)
                show_img = draw_ocr(image, boxes)  
                st.image(show_img)

                # OCR 결과를 세션 상태에 저장
                st.session_state.ocr_result = text_result

        st.success('OCR 완료!') 

        if 'llm_result' not in st.session_state:
            # OpenAI LLM 추론 및 결과 저장
            st.write('#### 결과값 생성 중')
            os.environ["openai_api_key"] = ""  # OpenAI 키 설정

            with st.spinner('Operating LLM ...'):
                llm = ChatOpenAI(
                    temperature=0.0,
                    max_tokens=128,
                    model_name="gpt-4o"
                )

                template = """
                영수증 데이터를 분석하여 다음 JSON 형식으로 정리하세요:
                결과만 json 형태로 보여주세요:
                ``` json
                {{
                    "store": ["상점명"],
                    "date": ["날짜"],
                    "total": ["총액"],
                }}
                ```
                아래는 분석해야 할 텍스트입니다:
                {ocr_result}
                """
                prompt = PromptTemplate.from_template(template).format(ocr_result=st.session_state.ocr_result)
                result_str = "".join([token.content for token in llm.stream(prompt)])

                # LLM 결과를 세션 상태에 저장
                st.session_state.llm_result = result_str

        st.success('LLM 완료!') 
        st.markdown(st.session_state.llm_result)

        # 파일 저장 및 다운로드
        file_types = ['None', 'xlsx', 'csv', 'md']
        choice = st.selectbox('다운받고 싶은 파일 유형을 선택하세요', file_types)

        if choice != "None" and st.session_state.get("llm_result"):
            data = json.loads(st.session_state.llm_result[7:][:-4])  # 결과 처리
            outputs = pd.DataFrame(data)

            if choice == "xlsx":
                df_xlsx = to_excel(outputs)
                st.download_button(
                    "xlsx 파일 다운로드", 
                    data=df_xlsx, 
                    file_name="영수증_결과.xlsx", 
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            elif choice == "csv":
                output_result = outputs.to_csv(index=False, encoding="utf-8")
                st.download_button(
                    "CSV 파일 다운로드", 
                    data=output_result, 
                    file_name="영수증_결과.csv", 
                    mime="text/csv"
                )
            elif choice == "md":
                st.markdown(outputs.to_markdown())

if __name__ == "__main__":
    main()
