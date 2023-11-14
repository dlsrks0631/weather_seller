# IQR (Interquartile Range) 규칙

## 개요

IQR 규칙은 데이터에서 이상치(outliers)를 식별하는 데 사용되는 통계적 방법으로 이 방법은 사분위수(quartiles)를 기반으로 하며, 데이터의 중간 50%를 나타내는 IQR 값에 기반하여 이상치의 범위를 정의한다.

## IQR 계산

IQR은 3사분위수(Q3)와 1사분위수(Q1)의 차이로 계산된다.

\[ \text{IQR} = Q3 - Q1 \]

여기서,
- **Q1 (1st Quartile)**: 데이터셋의 하위 25%
- **Q3 (3rd Quartile)**: 데이터셋의 상위 25%

## 이상치 식별

이상치는 다음과 같은 방법으로 식별됩니다:

\[ \text{Lower Bound} = Q1 - 1.5 \times \text{IQR} \]
\[ \text{Upper Bound} = Q3 + 1.5 \times \text{IQR} \]

여기서,
- 데이터 포인트가 Lower Bound보다 작거나 Upper Bound보다 큰 경우, 해당 데이터는 이상치로 간주된다.

## 예시 (Python)

```python
import numpy as np

data = np.array([1, 2, 3, 4, 5, 6, 7, 100])

Q1 = np.percentile(data, 25)
Q3 = np.percentile(data, 75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = data[(data < lower_bound) | (data > upper_bound)]
print("이상치:", outliers)
```

## 사용 분야

IQR 규칙은 데이터 과학, 금융 분석, 품질 관리 등 다양한 분야에서 이상치를 탐지하는 데 활용된다.

## 주의사항

- IQR 규칙은 대부분의 경우 잘 작동하지만, 모든 유형의 데이터 분포에 적합한 것은 아닙니다. 특히, 데이터가 심하게 비대칭적인 경우 오류를 일으킬 수 있다.
- 이상치가 항상 데이터 오류나 문제를 나타내는 것은 아니므로, 이상치를 제거하기 전에 데이터의 맥락을 고려해야 한다.
