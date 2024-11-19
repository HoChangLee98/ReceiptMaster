# import streamlit as st
# from stqdm import stqdm
# import os
# import pandas as pd
# from io import BytesIO
# import json
# from paddleocr import PaddleOCR
# import cv2
# import numpy as np
# from PIL import Image

# from langchain_openai import ChatOpenAI
# from langchain_core.prompts import PromptTemplate

# # from datetime import datetime
import streamlit as st
# from stqdm import stqdm
from paddleocr import PaddleOCR
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
import os
import pandas as pd
from io import BytesIO
import json
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
plt.rcParams['font.family'] ='Malgun Gothic'
plt.rcParams['axes.unicode_minus'] =False
os.environ["OPENAI_API_KEY"] = "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"  # OpenAI 키 설정

# 엑셀 저장 함수
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


# OCR 및 LLM 호출 함수
def process_image(image, ocr, llm, template):
    # OCR 실행
    ocr_result = ocr.ocr(image, cls=True)
    text_result = " ".join([line[1][0] for line in ocr_result[0]]).strip()[:500]  # 텍스트 길이 제한
    text_result = text_result.replace("\n", " ").strip()  # 줄바꿈 제거
    # st.write(f"### OCR 결과 (디버깅): {text_result}")

    # 2. LLM 프롬프트 생성
    try:
        prompt = PromptTemplate.from_template(template).format(ocr_result=text_result)
        # st.write(f"### LLM 프롬프트 (디버깅):\n{prompt}")  # 프롬프트 확인
    except KeyError as e:
        # st.error(f"프롬프트 생성 중 오류 발생: {e}")
        return {"store": "프롬프트 오류", "date": "N/A", "total": "N/A"}

     # 3. LLM 실행
    try:
        result_str = "".join([token.content for token in llm.stream(prompt)])
        # st.write(f"### LLM 응답 (디버깅): {result_str}")  # LLM 응답 확인

        # 4. JSON 응답 정제
        json_start = result_str.find("{")
        json_end = result_str.rfind("}") + 1
        if json_start == -1 or json_end == -1:
            raise ValueError("응답에서 JSON 형식을 찾을 수 없습니다.")

        clean_result_str = result_str[json_start:json_end]  # JSON 부분만 추출
        llm_result = json.loads(clean_result_str)

        return {
            "store": llm_result.get("store", "N/A"),
            "date": llm_result.get("date", "N/A"),
            "total": llm_result.get("total", "N/A"),
        }
    except json.JSONDecodeError as e:
        st.error(f"JSON 디코딩 오류 발생: {e}")
        return {"store": "JSON 오류", "date": "N/A", "total": "N/A"}
    except ValueError as e:
        st.error(f"LLM 응답 처리 오류: {e}")
        return {"store": "응답 오류", "date": "N/A", "total": "N/A"}


# 메인 이미지 처리 함수
def process_images(uploaded_files):
    ocr = PaddleOCR(
        use_angle_cls=False,
        lang='korean',
        use_gpu=False,
        det_model_dir="./ch_PP-OCRv4_det_infer",
        rec_model_dir="./korean_PP-OCRv3_rec_infer",
        rec_batch_num=10  # 배치 크기 증가로 성능 최적화
    )
    llm = ChatOpenAI(
        temperature=0.0,
        max_tokens=128,
        model_name="gpt-4"
    )
    template = """
    영수증 데이터를 분석하여 정확히 다음 JSON 형식으로만 반환하세요:
    ````json
    {{
        "store": "상점명",
        "date": "날짜 (YYYY-MM-DD)",
        "total": "총액 (숫자만 포함)"
    }}
    ```
    JSON 외에 다른 텍스트를 포함하지 마세요. 
    아래는 분석해야 할 텍스트입니다: {ocr_result} 
    """

    data = {"store": [], "date": [], "total": []}
    progress = st.progress(0)

    for i, img_file in enumerate(uploaded_files):
        st.write(f"{i+1}번째 이미지 처리 중...")

        # 이미지 읽기
        file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # 이미지 전처리
        image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        alpha, beta = 1.1, 5  # 밝기 및 대비 조정
        image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        image = cv2.fastNlMeansDenoisingColored(image, None, h=3, hColor=3, templateWindowSize=7, searchWindowSize=21)

        # OCR 및 LLM 호출
        with st.spinner(f"{i+1}번째 이미지에서 텍스트 추출 및 분석 중..."):
            result = process_image(image, ocr, llm, template)
            data["store"].append(result["store"])
            data["date"].append(result["date"])
            data["total"].append(result["total"])

        # 프로그레스 바 업데이트
        progress.progress((i + 1) / len(uploaded_files))

    return data


# Streamlit 메인 앱
def main():
    # st.title("OCR 및 LLM 기반 영수증 데이터 분석")
    img = Image.open('./sample-data/pig_img.png')
    # 경로에 있는 이미지 파일을 통해 변수 저장
    st.image(img)


    # 파일 업로드
    uploaded_files = st.file_uploader(
        "여러 영수증 사진을 업로드하세요", 
        type=["png", "jpg", "jpeg"], 
        accept_multiple_files=True
    )

    # 세션 상태 초기화
    if 'data' not in st.session_state:
        st.session_state.data = None

    if 'df_result' not in st.session_state:
        st.session_state.df_result = None

    if st.button("실행"):
        if uploaded_files:
            data = process_images(uploaded_files)
            st.session_state.data = data  # 세션 상태에 저장
            st.session_state.df_result = pd.DataFrame(data)  # 결과 DataFrame 저장
        else:
            st.error("파일을 업로드하세요!")

            # 파일 다운로드
            file_types = ['None', 'xlsx', 'csv', 'md']
            choice = st.selectbox("다운받고 싶은 파일 유형을 선택하세요", file_types)

    # 데이터가 있을 경우 결과 출력
    if st.session_state.df_result is not None:
        st.write("### 분석 결과")
        st.dataframe(st.session_state.df_result)
        
        df = pd.DataFrame(st.session_state.df_result)
        df["date"] = pd.to_datetime(df["date"])
        df["total"] = df["total"].astype(int)
        
        # 그래프 생성
        fig, ax = plt.subplots()
        ax.scatter(df["date"], df["total"], color="blue", s=0.5, alpha=0.7)
        ax.set_title("날짜 별 영수증 총 금액 그래프", fontsize=14)
        ax.set_xlabel("날짜", fontsize=12)
        ax.set_ylabel("결제 금액", fontsize=12)
        ax.grid(True)
        
        # Streamlit에 그래프 표시
        st.pyplot(fig)
        
        # 파일 다운로드
        file_types = ['None', 'xlsx', 'csv', 'md']
        choice = st.selectbox(
            "다운받고 싶은 파일 유형을 선택하세요", 
            file_types, 
            key="download_file_type"
        )

        if choice == "xlsx":
            df_xlsx = to_excel(st.session_state.df_result)
            st.download_button(
                "엑셀 파일 다운로드",
                data=df_xlsx,
                file_name="영수증_결과.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="xlsx_download"
            )
        elif choice == "csv":
            output_csv = st.session_state.df_result.to_csv(index=False, encoding="utf-8")
            st.download_button(
                "CSV 파일 다운로드",
                data=output_csv,
                file_name="영수증_결과.csv",
                mime="text/csv",
                key="csv_download"
            )
        elif choice == "md":
            st.markdown(st.session_state.df_result.to_markdown())



if __name__ == "__main__":
    main()