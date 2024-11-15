
import streamlit as st

import pandas as pd
from io import BytesIO
from pyxlsb import open_workbook as open_xlsb

from PIL import Image


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
    st.write('## 영수증에서 정보 추출하기')

    ## input 이미지 업로드 
    img_file = st.file_uploader('이미지를 업로드 하세요.', type=['png', 'jpg', 'jpeg'])

    # 파일이 업로드된 경우
    if img_file is not None:
        image = Image.open(img_file)

        ## PaddleOCR으로 텍스트 추출
        st.write('#### 텍스트 추출 중')
        
        ## Llama로 결과값 추출
        st.write('#### 결과값 생성 중')
                
        data = {
            "상품명": ["제품A", "제품B"],
            "가격": [10000, 20000],
            "수량": [2, 1]
        }

        ## 결과 xlsx, csv, md 중 하나로 추출
        file_types = ['None', 'xlsx', 'csv', 'md']
        choice = st.selectbox('다운받고 싶은 파일 유형을 선택하세요', file_types)

        if choice == "None":
            outputs = None
            
        elif choice == "xlsx":
            outputs = pd.DataFrame(data)
            st.dataframe(outputs)
            df_xlsx = to_excel(outputs)
            st.download_button(
                "xlsx 파일 다운로드", 
                data=df_xlsx, 
                file_name="영수증_결과.xlsx", 
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            
        ### 해보다가 빼기
        elif choice == "csv":
            outputs = pd.DataFrame(data)
            st.dataframe(outputs)
            output_result = outputs.to_csv(index=False, encoding="utf-8")
            st.download_button(
                "CSV 파일 다운로드", 
                data=output_result, 
                file_name="영수증_결과.csv", 
                mime="text/csv"
                )
            
        elif choice == "md":
            outputs = """```markdown
| 상품명 | 가격  | 수량 |
|--------|-------|------|
| 제품A  | 10000 | 2    |
| 제품B  | 20000 | 1    |
```"""
            st.markdown(outputs)

if __name__ == "__main__":
    main()        
