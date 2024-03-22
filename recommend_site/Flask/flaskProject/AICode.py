# 이 파일은 안 쓰는 파일입니다.

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import requests
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from collections import Counter
from sklearn.utils import resample
from torch.optim.lr_scheduler import StepLR

import json
import re

# 파일 경로 설정 (Flask 서버 내 적절한 경로로 설정)
FILE_PATH = 'data/전북 결제 날씨 통합데이터.csv'

# 데이터 로딩 함수 정의
def load_data():
    df = pd.read_csv(FILE_PATH)
    return df

# 전달 받은 데이터를 전처리하는 로직
def preprocess_input(df):
    """
        결측치를 처리하는 함수.

        :param df: 결측치를 확인하고 처리할 Pandas 데이터프레임
        :return: 결측치가 처리된 데이터프레임
    """
    # 결측치 확인
    missing_values = df.isnull().sum()

    # 결측치가 하나라도 있는 경우에만 처리를 수행합니다.
    if missing_values.any():
        # 최빈값으로 결측치를 대체하기 위한 imputer 생성
        imputer = SimpleImputer(strategy='most_frequent')

        # 전체 데이터셋에 imputer 적용
        df_filled = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

        # 대체된 데이터 프레임의 결측치를 다시 확인합니다.
        if df_filled.isnull().sum().any():
            print("결측치가 여전히 존재합니다.")
        else:
            print("모든 결측치가 처리되었습니다.")
        return df_filled
    else:
        print("결측치가 없습니다.")
        return df
# 이 함수는 다음과 같이 사용됩니다.
# df = pd.read_csv('path/to/data.csv')
# df_processed = handle_missing_values(df)


def filter_outliers(df, category_column, value_column):
    """
    IQR을 사용하여 데이터프레임 내 이상치를 필터링하는 함수.

    :param df: 이상치를 확인하고 처리할 Pandas 데이터프레임
    :param category_column: 카테고리를 나타내는 열의 이름
    :param value_column: 값(이상치를 확인할) 열의 이름
    :return: 이상치가 제거된 데이터프레임
    """
    filtered_df = pd.DataFrame()
    unique_categories = df[category_column].unique()

    for category in unique_categories:
        category_data = df[df[category_column] == category]
        Q1 = category_data[value_column].quantile(0.25)
        Q3 = category_data[value_column].quantile(0.75)
        IQR = Q3 - Q1

        # IQR 기준으로 이상치가 아닌 데이터만 필터링합니다.
        is_not_outlier = ~((category_data[value_column] < (Q1 - 1.5 * IQR)) |
                           (category_data[value_column] > (Q3 + 1.5 * IQR)))
        filtered_category_data = category_data[is_not_outlier]

        filtered_df = pd.concat([filtered_df, filtered_category_data], ignore_index=True)

    return filtered_df

# 사용 예시:
# df = pd.read_csv('path/to/data.csv')
# filtered_df = filter_outliers(df, '결제업종', '총결제금액')


# 정규화
def label_encode(df, column):
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    return df

def normalize_features(df, features_to_scale, string_features):
    """
    데이터프레임의 특성들을 레이블 인코딩하고 정규화하는 함수.

    :param df: 정규화할 Pandas 데이터프레임
    :param features_to_scale: 정규화를 적용할 수치형 특성 이름의 리스트
    :param string_features: 레이블 인코딩을 적용할 문자열 특성 이름의 리스트
    :return: 정규화된 데이터프레임
    """
    # 문자열 특성 레이블 인코딩 적용
    for feature in string_features:
        df = label_encode(df, feature)

    # MinMaxScaler 적용
    scaler = MinMaxScaler()
    df[features_to_scale] = scaler.fit_transform(df[features_to_scale])

    return df

# 사용 예시:
# df = pd.read_csv('path/to/data.csv')
# features_to_scale = ['성별', '연령대','총결제금액','결제개수','기온','강수량','적설량','전운량']
# string_features = ['식사시간', '날씨_구름', '날씨_비', '날씨_눈']
# df_normalized = normalize_features(df, features_to_scale, string_features)
# df_normalized.to_csv('normalized_data.csv', index=False)



# 날씨 정보를 API를 통해 가져오는 함수 정의
def get_weather_data(api_key, city, country_code):
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city},{country_code}&appid={api_key}&units=metric"
        response = requests.get(url)
        weather_data = response.json()

        # 기본 날씨 정보
        temperature = weather_data["main"]["temp"]
        rain = weather_data.get("rain", {}).get("1h", 0)  # 지난 1시간 동안의 강수량
        snow = weather_data.get("snow", {}).get("1h", 0)  # 지난 1시간 동안의 적설량
        clouds = weather_data["clouds"]["all"]  # 전운량
        weather_conditions = weather_data["weather"]
        weather_clouds = any(w["main"] == "Clouds" for w in weather_conditions)
        weather_rain = any(w["main"] == "Rain" for w in weather_conditions)
        weather_snow = any(w["main"] == "Snow" for w in weather_conditions)

        print(f"날씨 정보: {city} - 온도: {temperature}°C, 강수량: {rain}mm, 적설량: {snow}mm, 전운량: {clouds}%, 구름: {weather_clouds}, 비: {weather_rain}, 눈: {weather_snow}")

        return [temperature, rain, snow, clouds, weather_clouds, weather_rain, weather_snow]

    except Exception as e:
        print(f"Error getting weather data: {e}")
        return [0, 0, 0, 0, 0, False, False, False]

# 사용자 입력을 받는 함수 정의
def get_user_input():
    user_meal = input("식사 시간을 입력하세요 (새벽, 아침, 점심, 저녁, 야간): ")
    gender = int(input("성별을 입력하세요 (1: 남성, 2: 여성): "))
    age_range = int(input("연령대를 입력하세요 (예: 2, 3, 4...): "))
    residence_match = int(input("거주 유무를 입력하세요 (0: 일치하지 않음, 1: 일치): "))

    meal_time_mapping = {"새벽": 1, "아침": 2, "점심": 3, "저녁": 4, "야간": 5}
    meal_time = meal_time_mapping.get(user_meal.lower(), 0)
    return [meal_time, gender, age_range, residence_match]

# OpenWeatherMap API 키 설정
api_key = "d8bd115c7d5876a0e3275d8ed076a74b"

# 날씨 정보를 가져올 도시와 국가 코드
city = "Jeonju"
country_code = "KR"

# 날씨 정보를 가져옵니다.
weather_data = get_weather_data(api_key, city, country_code)

# 날씨 정보 출력
print(weather_data)

# 사용자 입력을 받습니다.
user_input = get_user_input()

# 날씨 정보와 사용자 입력을 결합하고 정규화합니다.
combined_input = user_input + weather_info
scaler = StandardScaler()
combined_input_scaled = scaler.fit_transform([combined_input])

# 모델 입력 크기를 설정합니다.
input_size = len(features_to_scale)

# 모델 아키텍처
class RecommendationModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RecommendationModel, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)

        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.5)

        self.layer3 = nn.Linear(hidden_size, hidden_size)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(0.5)

        self.layer4 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        x = self.layer2(x)
        x = self.relu2(x)
        x = self.dropout2(x)

        x = self.layer3(x)
        x = self.relu3(x)
        x = self.dropout3(x)

        x = self.layer4(x)
        return x

# 데이터 전처리 및 클래스 불균형 처리
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features_to_scale])
X_resampled, y_resampled = resample(X_scaled, df['결제업종'], random_state=42)

# 데이터셋 분리
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# LabelEncoder 인스턴스 생성
label_encoder = LabelEncoder()

# y_train을 정수 레이블로 변환
y_train_encoded = label_encoder.fit_transform(y_train)

# PyTorch 데이터셋 및 데이터 로더
train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train_encoded, dtype=torch.int64))
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(label_encoder.transform(y_test), dtype=torch.int64))
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 클래스 가중치 계산 (클래스 불균형 고려)
class_weights = 1.0 / torch.tensor(list(Counter(y_train_encoded).values()), dtype=torch.float32)

# 손실 함수 정의 (가중치 적용)
criterion = nn.CrossEntropyLoss(weight=class_weights)

# 모델 초기화 및 최적화 알고리즘 설정
input_size = X_train.shape[1]
output_size = len(label_encoder.classes_)
model = RecommendationModel(input_size, hidden_size=512, output_size=output_size)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Learning Rate 스케줄러 설정
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

# 조기 종료를 위한 변수 초기화
best_val_loss = float('inf')
patience = 15
wait = 0

num_epochs = 100
loss_threshold = 0.001  # 손실 임계값

# calculate_validation_loss 함수 정의
def calculate_validation_loss(model, val_loader, criterion):
    model.eval()  # 모델을 평가 모드로 설정
    total_loss = 0.0
    total_count = 0

    with torch.no_grad():
        for inputs, targets in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)
            total_count += inputs.size(0)

    model.train()  # 모델을 다시 훈련 모드로 설정
    return total_loss / total_count

# 모델 훈련
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    scheduler.step()

    if epoch % 10 == 0:
        # print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {total_loss / len(train_loader)}")
        pass

    if total_loss / len(train_loader) <= loss_threshold:
        # print(f"Training stopped as loss reached or went below {loss_threshold}")
        break

    # 검증 손실 계산 및 조기 종료 확인
    model.eval()
    val_loss = calculate_validation_loss(model, test_loader, criterion)
    if epoch % 10 == 0:
        # print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {total_loss / len(train_loader)}, Validation Loss: {val_loss}")
        pass

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        wait = 0
    else:
        wait += 1

    if wait >= patience:
        # print(f"Early stopping after {patience} epochs of no improvement in validation loss.")
        break

# 모델 저장
torch.save(model.state_dict(), 'model_weights.pth')

# 모델 평가
def evaluate_model(model, test_loader):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(targets.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    print(f'Accuracy: {accuracy * 100:.2f}%')
    print(f'Precision: {precision * 100:.2f}%')
    print(f'Recall: {recall * 100:.2f}%')
    print(f'F1 Score: {f1 * 100:.2f}%')

evaluate_model(model, test_loader)

# 추천 결제업종을 예측합니다.
cuisine_labels = ["한식", "중식", "일식", "양식"]
input_size = len(combined_input_scaled[0])
model = RecommendationModel(input_size=input_size, hidden_size=512, output_size=len(cuisine_labels))
combined_input_tensor = torch.tensor(combined_input_scaled, dtype=torch.float32)

with torch.no_grad():
    prediction = model(combined_input_tensor)
    probabilities = torch.nn.functional.softmax(prediction, dim=1).squeeze(0)

probabilities_percent = probabilities * 100
probabilities_percent_sorted, indices = torch.sort(probabilities_percent, descending=True)

# 결과 출력
print("추천 결제업종 및 확률:")
for i, idx in enumerate(indices):
    print(f"{cuisine_labels[idx]}: {probabilities_percent[idx].item():.2f}%")


# 폰트 경로 설정
font_path = '/content/drive/MyDrive/NanumBarunGothic.ttf'

# 폰트 추가
fm.fontManager.addfont(font_path)

# 기본 폰트로 설정
plt.rcParams['font.family'] = 'NanumBarunGothic'

# 추천 결제업종 및 확률
recommended_cuisines = [cuisine_labels[idx] for idx in indices]
recommended_probabilities = [probabilities_percent[idx].item() for idx in indices]

# 추천 결제업종 및 확률을 확률을 기준으로 내림차순으로 정렬
sorted_indices = sorted(range(len(recommended_probabilities)), key=lambda i: recommended_probabilities[i], reverse=True)
recommended_cuisines_sorted = [recommended_cuisines[i] for i in sorted_indices]
recommended_probabilities_sorted = [recommended_probabilities[i] for i in sorted_indices]