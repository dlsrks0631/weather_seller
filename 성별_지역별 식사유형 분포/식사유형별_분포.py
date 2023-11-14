# 지역별 연령대별 남자 양식, 중식, 한식 막대 그래프 코드
## 나중에 지역을 한 개 정할 때 근거로 쓰일 예정

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 폰트 경로 설정
font_path = r"C:\Users\user\Downloads\nanum-barun-gothic\NanumBarunGothic.ttf"

# 폰트 추가
fm.fontManager.addfont(font_path)

# 기본 폰트로 설정
plt.rcParams['font.family'] = 'NanumBarunGothic'

# CSV 파일을 불러옵니다. 이때 파일 경로는 실제 파일의 위치로 변경해야 합니다.
data = pd.read_csv(r"C:\Users\user\Desktop\캡스톤\롯데카드내역_지역(원본)\제주특별자치도_data.csv",encoding='utf-8')

# 성별이 남자(1)인 데이터 필터링
male_data = data[data['성별'] == 1]

# 결제업종이 양식, 중식, 한식인 데이터 필터링
filtered_data = male_data[male_data['결제업종'].isin(['양식', '중식', '한식', '일식'])]

# 연령대별로 그룹화하고, 결제업종별 결제개수 합산
grouped_data = filtered_data.groupby(['연령대', '결제업종'])['결제개수'].sum().unstack().fillna(0)


# 막대그래프 그리기
# 지역명 변경해야함
sns.set(style="whitegrid")
plt.figure(figsize=(12, 8))
grouped_data.plot(kind='bar', stacked=False)
plt.title('제주특별자치도 연령대별 남성의 양식, 중식, 한식, 일식식 결제건수', fontproperties=fm.FontProperties(fname=font_path))
plt.xlabel('연령대', fontproperties=fm.FontProperties(fname=font_path))
plt.ylabel('결제건수', fontproperties=fm.FontProperties(fname=font_path))
plt.legend(title='결제업종', prop=fm.FontProperties(fname=font_path))
plt.show()

# 이 밑에는 여자
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 폰트 경로 설정
font_path = r"C:\Users\user\Downloads\nanum-barun-gothic\NanumBarunGothic.ttf"

# 폰트 추가
fm.fontManager.addfont(font_path)

# 기본 폰트로 설정
plt.rcParams['font.family'] = 'NanumBarunGothic'

# CSV 파일을 불러옵니다. 이때 파일 경로는 실제 파일의 위치로 변경해야 합니다.
data = pd.read_csv(r"C:\Users\user\Desktop\캡스톤\롯데카드내역_지역(원본)\제주특별자치도_data.csv",encoding='utf-8')

# 성별이 여자(2)인 데이터 필터링
male_data = data[data['성별'] == 2]

# 결제업종이 양식, 중식, 한식인 데이터 필터링
filtered_data = male_data[male_data['결제업종'].isin(['양식', '중식', '한식', '일식'])]

# 연령대별로 그룹화하고, 결제업종별 결제개수 합산
grouped_data = filtered_data.groupby(['연령대', '결제업종'])['결제개수'].sum().unstack().fillna(0)


# 막대그래프 그리기
# 지역명 변경해야함
sns.set(style="whitegrid")
plt.figure(figsize=(12, 8))
grouped_data.plot(kind='bar', stacked=False)
plt.title('제주특별자치도 연령대별 여성의 양식, 중식, 한식, 일식 결제건수', fontproperties=fm.FontProperties(fname=font_path))
plt.xlabel('연령대', fontproperties=fm.FontProperties(fname=font_path))
plt.ylabel('결제건수', fontproperties=fm.FontProperties(fname=font_path))
plt.legend(title='결제업종', prop=fm.FontProperties(fname=font_path))
plt.show()
