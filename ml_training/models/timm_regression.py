import torch
import timm
from torch import nn
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# timm 모델을 불러오기만 하는 모델. regression 전용이다.
class TimmModel(nn.Module):
    def __init__(self, model_name, num_classes, in_chans, pretrained):
        super(TimmModel,self).__init__()
        self.algorithm = "regression"
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes, in_chans=in_chans)
    
    def forward(self, inputs):
        x = inputs[0].to(device)
        x = self.model(x)
        return x
    
    def getAlgorithm(self):
        return self.algorithm


def create_model(model_name, num_classes, in_chans, pretrained):
    model = TimmModel(model_name, num_classes, in_chans, pretrained)
    return model
