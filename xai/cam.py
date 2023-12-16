import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import timm
import cv2
import matplotlib.ticker as ticker
import torch.nn.functional as F
import argparse

from torchvision.transforms import ToTensor
from pathlib import Path
from PIL import Image
from torchvision import datasets, transforms

from pytorch_grad_cam import (
    GradCAM,
    HiResCAM,
    ScoreCAM,
    GradCAMPlusPlus,
    AblationCAM,
    XGradCAM,
    EigenCAM,
    FullGrad,
)
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import (
    show_cam_on_image,
    preprocess_image,
    deprocess_image,
)
from pytorch_grad_cam import GuidedBackpropReLUModel



class CAMVisualizer:
    # 모델과 이미지 크기, cuda 사용을 정의한다
    def __init__(self, model_url, img_url, img_size=448, use_cuda=False):
        self.model = self.load_model_from_url(model_url)
        self.img_url = img_url
        self.img_size = img_size
        self.use_cuda = use_cuda and torch.cuda.is_available()

    # 모델 경로를 통해 모델을 load한다.
    def load_model_from_url(self, model_url):
        # Assuming the model URL points to a PyTorch state dict
        model_state_dict = torch.hub.load_state_dict_from_url(model_url)
        model = create_model()  # Replace this with your actual model creation function
        model.load_state_dict(model_state_dict)
        if self.use_cuda:
            model = model.cuda()
        model.eval()
        return model

    # 이미지 경로를 통해 이미지를 읽어 정보를 반환한다.
    def load_image_from_url(self):
        response = requests.get(self.img_url)
        image = Image.open(BytesIO(response.content))
        return image
        
    # vit input에 맞게 tensor를 변경한다.
    def reshape_vit_transform(self, tensor, height=14, width=14):
        result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
        result = result.transpose(2, 3).transpose(1, 2)
        return result

    
        
    # vit 모델에서 xai image를 추출한다.
    # 이미지 데이터와 cam 종류를 parameter로 가진다.
    # target_classes를 None으로 주면 가장 스코어 값이 높은 xai 이미지를 반환한다. 여러 class를 설정하면 여러 class에 대한 xai 이미지가 반환된다.
    # 리턴값은 xai 이미지에 대한 데이터로 plt를 이용해 열 수 있다.
    def visualize_vit_model(self, image, cam_type, target_classes=None):
        transform = transforms.Compose([transforms.ToTensor()])
        input_tensor = transform(image).unsqueeze(0)

        if cam_type == "GradCAM":
            cam = GradCAM(
                self.model,
                target_layers=[self.model.blocks[-1].norm1],
                use_cuda=self.use_cuda,
                reshape_transform=self.reshape_vit_transform,
            )
        elif cam_type == "ScoreCAM":
            cam = ScoreCAM(
                self.model,
                target_layers=[self.model.blocks[-1].norm1],
                use_cuda=self.use_cuda,
                reshape_transform=self.reshape_vit_transform,
            )
        # Add more CAM types as needed.

        if target_classes is None:
            targets = None
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
            grayscale_cam = grayscale_cam[0, :]
            img_normalized = np.float32(image) / 255.0
            visualization = show_cam_on_image(
                img_normalized, grayscale_cam, use_rgb=True
            )
            return visualization.tolist()
        else:
            visualizations = []
            for target_class in target_classes:
                targets = [ClassifierOutputTarget(target_class)]
                grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
                grayscale_cam = grayscale_cam[0, :]
                img_normalized = np.float32(image) / 255.0
                visualization = show_cam_on_image(
                    img_normalized, grayscale_cam, use_rgb=True
                )
                visualizations.append(visualization.tolist())
            return visualizations
            
    # cnn 모델에서 xai 이미지를 추출한다.
    # 이미지 데이터와 cam 종류를 parameter로 가진다.
    # target_classes를 None으로 주면 가장 스코어 값이 높은 xai 이미지를 반환한다. 여러 class를 설정하면 여러 class에 대한 xai 이미지가 반환된다.
    # 리턴값은 xai 이미지에 대한 데이터로 plt를 이용해 열 수 있다.
    def visualize_cnn_model(self, img_path, cam_type, target_classes=None):
        img = Image.open(img_path)
        transform = transforms.Compose(
            [transforms.Resize((self.img_size, self.img_size)), transforms.ToTensor()]
        )
        input_tensor = transform(img).unsqueeze(0)
        img = img.resize((self.img_size, self.img_size))

        if cam_type == "GradCAM":
            cam = GradCAM(
                self.model,
                target_layers=[self.model.layer4[-1]],
                use_cuda=self.use_cuda,
            )
        elif cam_type == "ScoreCAM":
            cam = ScoreCAM(
                self.model,
                target_layers=[self.model.layer4[-1]],
                use_cuda=self.use_cuda,
            )
        # Add more CAM types as needed.

        if target_classes is None:
            targets = None
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
            grayscale_cam = grayscale_cam[0, :]
            img_normalized = np.float32(img) / 255.0
            visualization = show_cam_on_image(
                img_normalized, grayscale_cam, use_rgb=True
            )
            return visualization.tolist()
        else:
            visualizations = []
            for target_class in target_classes:
                targets = [ClassifierOutputTarget(target_class)]
                grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
                grayscale_cam = grayscale_cam[0, :]
                img_normalized = np.float32(img) / 255.0
                visualization = show_cam_on_image(
                    img_normalized, grayscale_cam, use_rgb=True
                )
                visualizations.append(visualization.tolist())
            return visualizations

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Visualize XAI images using CAM techniques.')
parser.add_argument('--image_path', type=str, required=True, help='Path to the input image.')
parser.add_argument('--model_path', type=str, required=True, help='Path to the model.')

args = parser.parse_args()

# Load the image
try:
    image = Image.open(args.image_path)
except Exception as e:
    print(f"Error opening image: {e}")
    exit(1)
    
# Instantiate the visualizer with the model path and image path
visualizer = CAMVisualizer(model_path=args.model_path, img_path=args.image_path, use_cuda=False)

# Choose the CAM technique and optional target classes (None for top predicted class)
cam_type = "GradCAM"
target_classes = None


visualization = visualizer.visualize_vit_model(image, cam_type, target_classes)
for vis in visualization:
    plt.figure()
    plt.axis("off")
    plt.imshow(vis)
    plt.show()
