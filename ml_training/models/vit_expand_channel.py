import torch
import torch.nn.functional as F
from torch import nn
import timm
from torchvision.models.feature_extraction import create_feature_extractor
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class LastModule(nn.Module):
    def __init__(self, in_chans, out_chans):
        super().__init__() 
        output_shape = 5
        self.fc1 = nn.Linear(in_chans, in_chans*2, bias=True)
        self.fc2 = nn.Linear(in_chans*2, out_chans, bias=True)

    def forward(self, x):
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        return x


class ECModel(nn.Module):
    def __init__(self, model_name, num_classes, in_chans):
        super(GCModel,self).__init__()
        self.algorithm = "regression"
        self.fc_in_chans = 0
        
        model_1 = timm.create_model(model_name, pretrained=True, num_classes=num_classes, in_chans=in_chans)
        self.model_1 = create_feature_extractor(model_1, return_nodes={"fc_norm":"out"})
        
        self.fc_in_chans += self.model_1.state_dict()[list(self.model_1.state_dict())[-1]].shape[-1]
        self.fc_layer = LastModule(self.fc_in_chans, num_classes)
    
    
    def forward(self, inputs):
        x = None
        for input in inputs:
            input = input.to(device)
            if x is None:
                x = input
            else:
                x = torch.concat([x,input],dim=1)
            
        output = self.model_1(x)['out']
        output = self.fc_layer(output)
        return output

    def getAlgorithm(self):
        return self.algorithm


def create_model(model_name, num_classes, in_chans, pretrained):
    model = ECModel(model_name, num_classes, in_chans)
    return model
