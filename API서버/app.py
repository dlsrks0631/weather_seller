from flask import Flask, send_from_directory, request, jsonify
import Recommend
from database import get_user_info  # 사용자 정보를 가져오는 함수
from datetime import datetime

app = Flask(__name__)


@app.route('/test.html')
def serve_static():
    return send_from_directory('static/html', 'test.html')


def get_current_meal_time():
    current_hour = datetime.now().hour
    if 6 <= current_hour < 12:
        return "아침"
    elif 12 <= current_hour < 15:
        return "점심"
    elif 15 <= current_hour < 20:
        return "저녁"
    elif 20 <= current_hour < 24 or 0 == current_hour:
        return "야간"
    else:
        return "새벽"


@app.route('/recommend', methods=['POST'])
def recommend():
    username = request.get_json().get('username')  # JSON 형식으로 데이터를 받습니다.
    print("Username : " + username)

    # 데이터베이스에서 사용자 정보를 가져옴
    user_info = get_user_info(username)

    # 반환된 user_info가 None인지 확인
    if user_info:
        # user_info가 None이 아니면, 언패킹을 수행
        gender, age = user_info

        # 현재 시간에 기반한 식사 시간 결정
        current_meal_time = get_current_meal_time()

        # gender 매핑
        gender_mapping = {'female': 2, 'male': 1}
        mapped_gender = gender_mapping.get(gender.lower(), 0)  # 없는 경우 기본값 0

        # age 매핑
        age_mapping = {'20s': 2, '30s': 3, '40s': 4, '50s': 5, '60s': 6}
        mapped_age = age_mapping.get(age, 0)  # 없는 경우 기본값 0

        # Recommend에서 모델 실행
        result = Recommend.run_model(current_meal_time, mapped_gender, mapped_age)

        print("result : " +result)
        # 결과를 springboot로 보냄
        return result

    else:
        # user_info가 None이면, 오류 메시지를 반환
        return jsonify({'error': 'User not found'}), 404


if __name__ == '__main__':
    app.run(debug=True)
