# Deeplant AWS web server
AWS Machine Learning EC2를 이용한 모델 배포

백엔드 서버 Flask 사용

# Start server
모델을 시작하고 로그를 남기기 위해 nohup을 이용
```
python3 nohup app.py
```

# Send request 로컬 -> 서버
로컬에서 서버로 이미지의 경롤르 보내서 결과를 받아 볼 수 있다.
이미지 경로는 AWS S3 버킷에 있는 이미지의 경로를 입력
```
python3 request.py --image_path "https://xai-deeplant-image.s3.ap-northeast-2.amazonaws.com/meat_image.jpg"
```

## 실행 결과 예시
```
Data sent               20:32:17.183380
Data received           20:32:23.587630
Grade: 1+
Marbling Score: 1.58
Color Score: 2.82
Texture Score: 2.82
Surface Moisture Score: 2.45
Total Score: 2.35
```
