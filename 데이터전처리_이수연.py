import pandas as pd
from prettytable import PrettyTable

# 엑셀 파일을 불러옵니다. 파일 경로를 실제 파일 경로로 수정하세요.
excel_file = r'C:\Users\mypc\Desktop\데이터.xlsx'
df = pd.read_excel(excel_file)

# 필요한 열만 선택합니다.
selected_columns = ['결제시간', '광역시군구', '시군구', '남녀구분', '연령대', '결혼여부', '자녀여부', '유입여부', '업종 중분류', '결제금액', '결제건수']
df = df[selected_columns]

# '광역시군구' 열에 '전라북도'를 추가합니다.
df['광역시군구'] = '전라북도'

# '결제시간'을 시간대로 분류합니다.
df['결제시간'] = df['결제시간'].apply(lambda x: '12-14시' if '12-14' in x else '15-17시')

# 시간대 분류 함수를 정의합니다.
def classify_time(row):
    if '12-14시' in row['결제시간']:
        return '12시-14시'
    elif '15-17시' in row['결제시간']:
        return '15시-17시'
    else:
        return '기타'

# '기준년월일' 열을 시간대로 분류하여 '시간대' 열을 추가합니다.
df['시간대'] = df.apply(classify_time, axis=1)

# 업종 중분류를 순서대로 정의합니다.
category_order = ['한식', '중식', '일식', '양식', '카페/디저트', '기타일반음식', '유통업', '숙박']

# 업종 중분류를 Categorical 데이터 타입으로 변환하고 정렬합니다.
df['업종 중분류'] = pd.Categorical(df['업종 중분류'], categories=category_order, ordered=True)

# '시군구', '남녀구분', '연령대', '결혼여부', '자녀여부', '유입여부' 순서로 정렬합니다.
# '결제금액'과 '결제건수'를 오름차순으로 정렬합니다.
df = df.sort_values(by=['시군구', '남녀구분', '연령대', '결혼여부', '자녀여부', '유입여부', '결제금액', '결제건수'], ascending=[False, True, True, True, True, False, True, True])

# '시간대' 열을 삭제합니다.
df = df.drop(columns=['시간대'])

# PrettyTable을 사용하여 표 형태로 출력합니다.
table = PrettyTable()
table.field_names = df.columns
table.max_width = 10  # 표의 최대 너비를 조절합니다.
for row in df.itertuples(index=False):
    table.add_row(row)

print(table)





