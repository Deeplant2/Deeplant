import requests
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


def send_request(image_path):
    url = "http://52.79.161.151:5000/predict"
    data = {"image_path": image_path}
    print(f"Data sent \t\t{datetime.now().time()}")
    response = requests.post(url, json=data)
    print(f"Data received \t\t{datetime.now().time()}")
    try:
        result = response.json()
        # print(result)
        output = result.get("xai_gradeNum_imagePath")
    except ValueError as e:
        print("Error: Invalid JSON response from the server.")
        print("Exception:", e)
        return None

    if response.status_code == 200:
        grade = response.json()["gradeNum"]
        marbling = response.json()["marbling"]
        color = response.json()["color"]
        texture = response.json()["texture"]
        surfaceMoisture = response.json()["surfaceMoisture"]
        overall = response.json()["overall"]
        return grade, marbling, color, texture, surfaceMoisture, overall
    else:
        print("Error: Failed to get a valid response from the server.")


# Usage example:
image_path = "https://xai-deeplant-image.s3.ap-northeast-2.amazonaws.com/meat_image.jpg"

## grade, marbling, color, texture, surfaceMoisture, overall
grade, marbling, color, texture, surfaceMoisture, overall = send_request(
    image_path
)

print(grade, marbling, color, texture, surfaceMoisture, overall)
