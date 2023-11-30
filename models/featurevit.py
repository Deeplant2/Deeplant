import torch
import torch.nn.functional as F
from torch import nn
import timm
from torchvision.models.feature_extraction import create_feature_extractor

device = "cuda" if torch.cuda.is_available() else "cpu"


class CombinedModel(nn.Module):
    def __init__(self):
        super(CombinedModel, self).__init__()
        self.algorithm = "regression"
        self.cnn_model = timm.create_model(
            "efficientnetv2_rw_s.ra2_in1k", pretrained=True, features_only=True
        )
        self.vit_model = timm.create_model(
            "vit_base_patch16_224.augreg2_in21k_ft_in1k",
            pretrained=True,
            num_classes=5,
            exportable=True,
            in_chans=272,
            img_size=224,
        )
    def forward(self, x):
        x = x.to(device)
        cnn_features = self.cnn_model(x)
        upscaled_feature_map = torch.nn.functional.interpolate(
            cnn_features[4], size=(224, 224), mode="bilinear", align_corners=False
        )

        torch_img = upscaled_feature_map.clone().detach()
        vit_output = self.vit_model(torch_img)
        return vit_output

    def getAlgorithm(self):
        return self.algorithm


def create_model():
    model = CombinedModel()
    return model
