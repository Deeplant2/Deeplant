import torch
import torch.nn.functional as F
from torch import nn
import timm
from torchvision.models.feature_extraction import create_feature_extractor
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class FeatureViT(nn.Module):
    def __init__(self):
        super(CombinedModel, self).__init__()
        self.algorithm = "regression"
        self.cnn_model = timm.create_model('efficientnetv2_rw_s.ra2_in1k', pretrained=True, features_only=True)
        self.vit_model = timm.create_model(
            "vit_base_patch32_clip_448.laion2b_ft_in12k_in1k",
            pretrained=True,
            num_classes=5,
            exportable=True,
            in_chans=24 #ViT 모델의 input channel을 CNN featuremap에 맞춰서 확장
        )

    def forward(self, x):
        x[0] = x[0].to(device)
        cnn_features = self.cnn_model(x[0])

        layer = 1 #1-5 사이에 CNN featuremap을 추출하는 layer 선택

        # featuremap (448,448) 크기로 업스케일링 
        upscaled_feature_map = torch.nn.functional.interpolate(cnn_features[layer-1], size=(448, 448), mode='bilinear', align_corners=False)
        torch_img = upscaled_feature_map.clone().detach()

        # ViT모델에 featuremap 전달
        vit_output = self.vit_model(torch_img)
        return vit_output

    def getAlgorithm(self):
        return self.algorithm
def create_model():
    model = FeatureViT()
    return model
