import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from datetime import datetime

def process_data(data, region_name):
    print(f"Processing data for {region_name}...")

    # 결측치 처리
    imputer = SimpleImputer(strategy="median")
    data["sl_am"] = imputer.fit_transform(data[["sl_am"]])

    # 이상치 처리
    z_scores = np.abs((data["sl_am"] - data["sl_am"].mean()) / data["sl_am"].std())
    data = data[(z_scores < 3)]

    # 정규화
    scaler = StandardScaler()
    data[["sl_am", "sl_ct"]] = scaler.fit_transform(data[["sl_am", "sl_ct"]])

    return data

def main():
    print("Start processing:", datetime.now())

    # 지역 목록
    regions = [
        "전라북도", "강원도", "경기도",
        "부산광역시", "서울특별시", "인천광역시",
        "제주특별자치도"
    ]

    # 데이터 불러오기
    data_list_by_region = {region: [] for region in regions}
    
    # 파일 경로는 각자 파일에 맞게 바꿔야합니다
    chunk_iter = pd.read_csv("/Users/kimjehyeon/Desktop/2023-10-10-롯데카드_소비_데이터.csv", encoding='CP949', chunksize=1000000)

    for chunk in chunk_iter:
        for region in regions:
            filtered_chunk = chunk[chunk['ana_mgpo_nm'] == region]
            data_list_by_region[region].append(filtered_chunk)

    processed_data_by_region = {}

    # 각 지역 데이터 처리
    for region, data_list in data_list_by_region.items():
        regional_data = pd.concat(data_list, axis=0)
        processed_data_by_region[region] = process_data(regional_data, region)

    # 결과 출력
    for region in regions:
        print(f"{region} 데이터 상위 5개 행:")
        print(processed_data_by_region[region].head())
        print("\n")

    print('Finished Processing:', datetime.now())

    # 결과 반환 (필요한 경우)
    return processed_data_by_region

if __name__ == '__main__':
    processed_data_by_region = main()
