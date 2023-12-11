import torch
import timm
import torch.nn.functional as F
from einops import rearrange
from torch import nn
from torchvision.models.feature_extraction import create_feature_extractor 
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class LastModule(nn.Module):
    def __init__(self):
        super(LastModule, self).__init__()
        input_shape = 5
        output_shape = 5

        self.fc0 = nn.Linear(input_shape,output_shape*2, bias = False)
        self.fc1 = nn.Linear(output_shape*2, output_shape,bias = False)
    def forward(self, x):
        x=self.fc0(x)
        x=F.relu(x)
        x=self.fc1(x)
        return x


class CoatNet(nn.Module):
    def __init__(self):
        super(CoatNet, self).__init__()
        self.algorithm = "regression"
        
        model_name = "coatnet_small"
        num_classes = 5

        if model_name =='coatnet_regular':
            temp_model = timm.create_model('coat_mini',pretrained=True, exportable =True, num_classes=num_classes)
            #feature={'norm2.norm2_weight' : 'out'}
        if model_name =='coatnet_small':
            temp_model = timm.create_model('coat_small',pretrained=True, exportable =True, num_classes=num_classes)
            #feature ={'norm2.norm2_weight' : 'out'}

        if model_name == 'coatnet_regualr':
            temp_model = timm.create_model('coat_3_rw_224',pretrained=True, exportable =True, num_classes=num_classes)
            #feature={'norm.permute_1' : 'out'}

        #extractor = create_feature_extractor(temp_model, return_nodes = feature)

        #self.models.append(extractor.to(device))
        self.models= temp_model.to(device)
        self.lastfc_layer = LastModule()

    def forward(self, inputs):
        x = None
        for input in inputs:
            input = input.to(device)
            if x is None:
                x = input
            else:
                x = torch.concat([x,input],dim=1)
        output = self.models(x)
        output = self.lastfc_layer(output)
        return output
    
    def getAlgorithm(self):
        return self.algorithm


def create_model():
    model = CoatNet()
    return model
