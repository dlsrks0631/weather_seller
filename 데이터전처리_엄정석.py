import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

#요구분석서에 있는 최소한의 정도를 구현하였습니다.
#데이터 전처리는 데이터 분석만 해도 결과보고서에 충분히 작성 가능할 것 같습니다.
#추후 신경망 모델, 추천 알고리즘을 공부한 뒤, 분석된 결과를 이용하여 사용 가능할 것 같습니다.

#코드를 참고하시면서, 다음을 생각해주시면 좋을 것 같습니다.
#  1. 추가해야 하는 것: 중복 데이터? 구간화?
#  2. 금융 정보와 비금융 정보에 해당하는 각각의 열 분류
#  3. 결측치, 이상치, 정규화 결과 보고
#  4. (추후 예정) PyTorch에서 배울 수 있는 신경망 모델, 추천 알고리즘에 대해 입출력 연관시키기

#고민사항을 수정하거나 추가하셔도 상관 없습니다. 
#여기에서 생기는 모든 질문은 수합하여 멘토님께 드릴 예정입니다.
#회의는 시험을 최대한 피하겠습니다. 혹시라도 시험 일정이 잠정적으로라도 결정되면 단톡방에 알려주세요~.

# 데이터 불러오기
data = pd.read_excel('2023-09-26-카드결제데이터샘플2.xlsx')

# 1. 결측치 확인
# 결측치? -> 데이터에 값이 없는 것 (Null 상태)
# 분석 결과: 샘플 데이터에서 결측치는 없다.
print("전체 데이터 건수: ", len(data))
print("컬럼별 결측치 개수")
print(len(data) - data.count())

# 만약 결측치가 있다면? -> 아래의 작업을 수행하면 된다.
# data = data.drop('시군구', axis=1)
# * data.drop(index = ?, axis = ?) // index: 행, axis: 열
# data.head()
# df.loc["row", "column"] -> 색인 위치 확인

# 2. 이상치 확인: Z점수 확인
z_scores = (data['결제금액'] - data['결제금액'].mean()) / data['결제금액'].std()
print("평균: ", data['결제금액'].mean(), "  표준편차: ", data['결제금액'].std())
print(data[(z_scores > 0)])

#+추가: 사분위수 확인
Q1 = data['결제금액'].quantile(.25)
Q3 = data['결제금액'].quantile(.75)
IQR = Q3 - Q1
print("IQR 값: ", IQR)

#사분위수 출력
fig, ax = plt.subplots()
ax.boxplot(data['결제금액'])
plt.show()

# 3. 정규화
#표준화(Standardization): Z-score
print("Z점수 결과(상위 5개)")
z_scores = (data['결제금액'] - data['결제금액'].mean()) / data['결제금액'].std()
print("평균: ", data['결제금액'].mean(), "  표준편차: ", data['결제금액'].std())
print(data[(z_scores > 0)].head())

#Min-Max Scaling: 데이터의 최솟값은 0, 최댓값은 1로 변환
print("Min-Max Scaling 결과")
min_max_scaling = (data['결제금액'] - data['결제금액'].min())/(data['결제금액'].max() - data['결제금액'].min())
print(min_max_scaling.head())

#측정치 설명
print("종합 결과")
print(data.describe())


#참고문헌: 
# https://iambeginnerdeveloper.tistory.com/30
# https://sosomemo.tistory.com/34
# ChatGPT


#추가 의견은 여기에 언제든 이야기해주세요~
# ================================================================================================================





# ================================================================================================================