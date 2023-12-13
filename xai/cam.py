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
    def __init__(self, model, img_size=448, use_cuda=False):
        self.model = model
        self.img_size = img_size
        self.use_cuda = use_cuda
        
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

# How to use
visualizer = CAMVisualizer(model=vit_model, img_size=448, use_cuda=False)

# Load an image you want to visualize
image = Image.open("your_image.jpg")

# Choose the CAM technique (e.g., "GradCAM") and optional target classes (None for top predicted class)
# GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
cam_type = "GradCAM"
target_classes = (None)

visualization = visualizer.visualize_vit_model(image, cam_type, target_classes)
for vis in visualization:
    plt.figure()
    plt.axis("off")
    plt.imshow(vis)
    plt.show()
