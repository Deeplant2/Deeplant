import requests
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


def send_request(image_path):
    url = "http://13.209.0.181:5000/predict"
    data = {"image_path": image_path}
    print(f"Data sent \t\t{datetime.now().time()}")
    response = requests.post(url, json=data)
    print(f"Data received \t\t{datetime.now().time()}")

    print(response.json())

    if response.status_code == 200:
        grade = response.json()["gradeNum"]
        marbling = response.json()["marbling"]
        color = response.json()["color"]
        texture = response.json()["texture"]
        surfaceMoisture = response.json()["surfaceMoisture"]
        marbling = response.json()["marbling"]
        overall = response.json()["overall"]
        return grade, marbling, color, texture, surfaceMoisture, marbling, overall
    else:
        print("Error: Failed to get a valid response from the server.")


# Usage example:
image_path = "https://deep-plant-image.s3.ap-northeast-2.amazonaws.com/sensory_evals/f3ddad6010f7-1.png"
# image_path = "https://deep-plant-image.s3.ap-northeast-2.amazonaws.com/sensory_evals/QC_cow_segmentation_1%2B_069481.jpg"

## grade, marbling, color, texture, surfaceMoisture, marbling, overall
grade, marbling, color, texture, surfaceMoisture, marbling, overall = send_request(
    image_path
)


{
    "color": 0.9657164812088013,
    "gradeNum": "1+",
    "marbling": 2.1435980796813965,
    "overall": 0.22654150426387787,
    "surfaceMoisture": 0.4809803366661072,
    "texture": 0.8446391224861145,
    "xai_gradeNum_imagePath": "https://xai-deep-plant-image.s3.ap-northeast-2.amazonaws.com/grade_xai.png",
    "xai_imagePath": "https://xai-deep-plant-image.s3.ap-northeast-2.amazonaws.com/sensory_xai.png",
}