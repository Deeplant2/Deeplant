# Deeplant AWS web server
AWS Machine Learning EC2를 이용한 모델 배포

백엔드 서버 Flask 사용

# Start server
모델 서버를 시작하고 로그를 남기기 위해 nohup을 이용
```
python3 nohup app.py
```

# Send request 로컬 -> 서버
로컬에서 서버로 이미지의 경로를 보내서 결과를 받아 볼 수 있다.
이미지 경로는 AWS S3 버킷에 있는 이미지의 경로를 입력한다.
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

## 파일 설명
### app.py
flask를 구동하는 파일이다. 데이터를 예측할 이미지의 경로가 들어오면 등급 데이터와 관능 데이터와 등급에 대한 XAI 이미지 경로와 관능에 대한 XAI 이미지 경로가 반환된다.

### cam.py
XAI 이미지를 생성하는 파일이다. 모델별로 XAI 이미지를 생성하는 방법이 다르기 때문에 모델별로 반환하는 함수가 만들어져있다. 

### models
models 디렉토리는 모델 코드가 포함되어있는 디렉토리이다.

### image
예시 이미지가 들어있는 폴더이다.

### pytorch_grad_cam
XAI 이미지를 만들기 위한 코드가 포함된 디렉토리이다. 해당 코드는 https://github.com/jacobgil/pytorch-grad-cam에서 사용한 코드이다.
