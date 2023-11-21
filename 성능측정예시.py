from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# 예시 데이터 (실제 데이터로 교체 필요)
actual_choices = [...]  # 사용자가 실제로 선택한 업종 목록
predicted_choices = [...]  # 시스템이 추천한 업종 목록

# 성능 지표 계산
accuracy = accuracy_score(actual_choices, predicted_choices)
precision = precision_score(actual_choices, predicted_choices, average='macro')
recall = recall_score(actual_choices, predicted_choices, average='macro')
f1 = f1_score(actual_choices, predicted_choices, average='macro')

# 결과 출력
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")

# 참고로, 이건 GPT에게서 가져온 것입니다. 정확하지 않을 수 있는 점 참고 바랍니다.

import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 예시 데이터 (실제 데이터로 교체 필요)
# 이 예시에서는 각 업종별로 사용자가 실제로 선택한 업종과 시스템이 추천한 업종이 있는 것으로 가정합니다.
actual_choices = ['한식', '중식', '한식', '일식', '양식']  # 사용자가 실제로 선택한 업종
predicted_choices = ['한식', '일식', '한식', '일식', '중식']  # 시스템이 추천한 업종

# 성능 지표 계산
accuracy = accuracy_score(actual_choices, predicted_choices)
precision = precision_score(actual_choices, predicted_choices, average='macro', labels=np.unique(predicted_choices))
recall = recall_score(actual_choices, predicted_choices, average='macro', labels=np.unique(predicted_choices))
f1 = f1_score(actual_choices, predicted_choices, average='macro', labels=np.unique(predicted_choices))

# 결과 출력
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")

# average='macro'는 다중 클래스 문제에 적합한 평균 방식을 사용
# labels=np.unique(predicted_choices)는 모든 예측된 레이블에 대해 계산을 수행하도록 지정
