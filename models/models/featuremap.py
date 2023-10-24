import torch
import torch.nn.functional as F
from torch import nn
import timm
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision import transforms
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class LastModule(nn.Module):
    def __init__(self, input_shape):
        super().__init__() 
        output_shape = 5
        self.fc1 = nn.Linear(input_shape, input_shape*2, bias=True)
        self.fc2 = nn.Linear(input_shape*2, output_shape, bias=True)

    def forward(self, x):
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        return x

class ViTCnnModel(nn.Module):
    def __init__(self):
        super(ViTCnnModel,self).__init__()
        self.algorithm = "regression"
        self.fc_input_shape = 0
        model_1 = timm.create_model("vit_base_patch32_clip_448.laion2b_ft_in12k_in1k", pretrained=True, num_classes=5, in_chans=17)
        model_2 = timm.create_model("resnetrs152.tf_in1k", pretrained=True, num_classes=5)
        
        self.model = create_feature_extractor(model_1, return_nodes={"fc_norm":"out"})
        self.backbone = create_feature_extractor(model_2, return_nodes={"layer1":"out"})
        self.conv2d = nn.Conv2d(in_channels=256,out_channels=16, kernel_size=1, stride=1, bias=True)
        
        self.fc_input_shape += self.model.state_dict()[list(self.model.state_dict())[-1]].shape[-1]
        self.fc_layer = LastModule(self.fc_input_shape)
        
        self.transform = transforms.Resize([448,448])
    
    def forward(self, inputs):
        input_1 = inputs[0].to(device) #img
        input_2 = inputs[1].to(device) #graph
        
        
        x = self.backbone(input_1)['out']
        x = self.transform(x)
        x = self.conv2d(x)
        x = torch.concat([x,input_2],dim=1)
        x = self.model(x)['out']
        x = self.fc_layer(x)
        return x

    def getAlgorithm(self):
        return self.algorithm


def create_model():
    model = ViTCnnModel()
    return model
