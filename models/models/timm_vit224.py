import torch
import torch.nn.functional as F
from torch import nn
import timm
from torchvision.models.feature_extraction import create_feature_extractor
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class TimmModel(nn.Module):
    def __init__(self):
        super(TimmModel,self).__init__()
        self.algorithm = "regression"
        self.model = timm.create_model("vit_base_patch16_224.augreg_in21k_ft_in1k", pretrained=True, num_classes=1)
    def forward(self, inputs):
        input = inputs[0].to(device)
        output = self.model(input)
        
        return output

    def getAlgorithm(self):
        return self.algorithm

def create_model():
    model = TimmModel()
    return model
