import torch
import torch.nn.functional as F
from torch import nn
import timm
from torchvision.models.feature_extraction import create_feature_extractor
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class FCModule(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, input_dim*2, bias=True)
        self.fc2 = nn.Linear(input_dim*2, output_dim, bias=True)

    def forward(self, x):
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        return x

class test_model(nn.Module):
    def __init__(self):
        super(test_model,self).__init__()
        self.algorithm = "regression"
        self.fc_input_shape = 0
        
        temp_model = timm.create_model("vit_base_patch32_clip_448.laion2b_ft_in12k_in1k", pretrained=True, num_classes=5)
        features={'fc_norm':'out'} # fc_norm, global_pool.flatten 
        feature_extractor = create_feature_extractor(temp_model, return_nodes = features)

        
        self.model_1 = feature_extractor.to(device)
        self.fc_input_shape += self.model_1.state_dict()[list(self.model_1.state_dict())[-1]].shape[-1]
        self.fc1 = FCModule(self.fc_input_shape,5)
        self.fc2 = FCModule(9,5)

    def forward(self, inputs):
        img = inputs[0].to(device)
        grade = inputs[1].to(device)
        x = self.model_1(img)['out']
        x = self.fc1(x)
        x = torch.concat([x,grade],dim=-1)
        x = self.fc2(x)
            
        return x

    def getAlgorithm(self):
        return self.algorithm

def create_model():
    model = test_model()
    return model
