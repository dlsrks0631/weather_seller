# RobustScaler

## 개요

`RobustScaler`는 데이터 전처리 과정에서 사용되는 스케일링 기법 중 하나입니다. 이 기법은 특히 이상치(outliers)에 강한 내성을 가지고 있어, 이상치가 포함된 데이터셋에 적용할 때 유용하다.

## 작동 원리

`RobustScaler`는 각 특성(feature)의 중앙값(median)을 빼고, 사분위 범위(Interquartile Range, IQR)로 나누어 데이터를 스케일링한다. 이렇게 실행할 경우 이상치의 영향을 최소화할 수 있습니다.

수식으로 표현하면 다음과 같습니다:

\[ \text{Scaled} = \frac{\text{X} - \text{Median}}{\text{IQR}} \]

여기서,
- **X**: 원래 데이터 포인트
- **Median**: 해당 특성의 중앙값
- **IQR**: 1사분위수(Q1)와 3사분위수(Q3)의 차이 (Q3 - Q1)

## 사용 예시 (Python)

```python
from sklearn.preprocessing import RobustScaler
import numpy as np

data = np.array([[1, 2, 3],
                 [4, 5, 6],
                 [7, 100, 9]])

scaler = RobustScaler()
scaled_data = scaler.fit_transform(data)

print(scaled_data)
```

## 장점 및 적용 분야

- **이상치 영향 최소화**: 이상치에 덜 민감하므로, 다른 스케일링 방법보다 이상치가 있는 데이터셋에 적합하다.
- **유연성**: 다양한 데이터셋에 적용 가능하며, 특히 금융 분야에서 이상치가 자주 발생하는 데이터에 유용하다.

## 주의사항

- `RobustScaler`는 데이터의 분포를 유지하면서 이상치의 영향을 줄이지만, 모든 데이터셋에 최적인 것은 아니다. 따라서 데이터의 특성을 고려하여 적절한 스케일링 방법을 선택해야 한다.
