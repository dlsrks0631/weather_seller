# database.py
import pymysql

# 데이터베이스 연결 설정
db_config = {
    'host': 'localhost',
    'user': 'boot',
    'password': '1234',
    'db': 'bootstudy',
    'charset': 'utf8mb4'
}

def get_user_info(username):
    # 데이터베이스 연결
    connection = pymysql.connect(**db_config)

    try:
        with connection.cursor() as cursor:
            # 특정 필드만 가져오는 쿼리 (예: gender와 age)
            sql = "SELECT gender, age FROM member WHERE username = %s"
            cursor.execute(sql, (username,))
            user_info = cursor.fetchone()
            # user_info => 튜플 ex) ('female', '20s')
            return user_info
    finally:
        connection.close()
