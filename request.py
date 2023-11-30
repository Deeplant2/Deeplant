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
        # Response is valid
        grade = result["gradeNum"]
        marbling = round(result["marbling"],2)
        color = round(result["color"],2)
        texture = round(result["texture"],2)
        surfaceMoisture = round(result["surfaceMoisture"],2)
        total = round(result["overall"],2)
        return grade, marbling, color, texture, surfaceMoisture, total
    except ValueError as e:
        print("Error: Invalid JSON response from the server.")
        print("Exception:", e)
        return None

def main():
    parser = argparse.ArgumentParser(description='Send image path as a request to the server.')
    parser.add_argument('--image_path', type=str, default="https://xai-deeplant-image.s3.ap-northeast-2.amazonaws.com/meat_image.jpg", help='Path to the image')
    args = parser.parse_args()

    grade, marbling, color, texture, surfaceMoisture, total = send_request(args.image_path)

    print(f"Grade: {grade}")
    print(f"Marbling Score: {marbling}")
    print(f"Color Score: {color}")
    print(f"Texture Score: {texture}")
    print(f"Surface Moisture Score: {surfaceMoisture}")
    print(f"Total Score: {total}")

if __name__ == '__main__':
    main()
