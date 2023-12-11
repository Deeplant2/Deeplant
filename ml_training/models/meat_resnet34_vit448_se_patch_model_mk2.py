import torch
import timm
import torch.nn.functional as F
from einops import rearrange
from torch import nn
from torchvision.models.feature_extraction import create_feature_extractor 
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# +
class LastModule(nn.Module):
    def __init__(self):
        super(LastModule, self).__init__()
        input_shape = 768
        output_shape = 5
        
#         mid_shape = int(input_shape * 0.75)
#         last_shape= int(input_shape*0.25)

#         mid_shape = int(input_shape * 2)
#         last_shape= int(input_shape*2 * 0.75)

        self.fc0 = nn.Linear(input_shape,1024, bias = True)
        self.fc1 = nn.Linear(1024, 2048, bias = True)
        self.fc2 = nn.Linear(2048, output_shape, bias =True)
    def forward(self, x):
        #(b,197,768) -> (768)
        x = x[:, 0, :]
        
        x=self.fc0(x)
        x=F.relu(x)
        x=self.fc1(x)
        x=F.relu(x)
        x=self.fc2(x)
        
        return x


# -

class InterMediateConv(nn.Module):
    def __init__(self, inp=128, oup=768, expansion=0.25):
        super(InterMediateConv,self).__init__()
        #input dimension change
        self.conv1 = nn.Conv2d(inp, oup, 1, 2)
        
        #SE block
        self.avg_pool  = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
        nn.Linear(oup, int(oup * expansion), bias= False),
        nn.GELU(),
        nn.Linear(int(oup * expansion), oup, bias= False),
        nn.Sigmoid())
    
    def forward(self, x):
        #(b, 128, 56, 56) -> (b, 256, 28, 28) 
        x = self.conv1(x)
        
        #SE block
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b,c)
        y = self.se(y).view(b,c,1,1)
        
        x= x*y
        
        #stacking 
        x1=x[:,   :256, :]
        x2=x[:,256:512, :]
        x3=x[:,512:768, :]
        
        x1 = rearrange(x1, 'b (c1 c2) h w -> b 1 (c1 h) (c2 w)', c1=16)
        x2 = rearrange(x2, 'b (c1 c2) h w -> b 1 (c1 h) (c2 w)', c1=16)
        x3 = rearrange(x3, 'b (c1 c2) h w -> b 1 (c1 h) (c2 w)', c1=16)
        
        x = torch.cat([x1,x2,x3],dim=1)
        return x


class Resnet34_VIT448_SE(nn.Module):
    def __init__(self):
        super(Resnet34_VIT448_SE, self).__init__()
        self.algorithm = "regression"
        
        model_name = "resnet34_vit448_se_patch"
        num_classes = 1
        temp_model_c = timm.create_model("resnet34.a1_in1k", pretrained=True, num_classes = 1, exportable=True)
        feature_c={'layer2.3.add':'out'}
        
        temp_model_v = timm.create_model('vit_base_patch32_clip_448.laion2b_ft_in12k_in1k',pretrained=True, num_classes=1,exportable=True)
        feature_v = {"blocks.1.add":"out"}
        
        extractor_c = create_feature_extractor(temp_model_c, return_nodes = feature_c)
        extractor_v = create_feature_extractor(temp_model_v, return_nodes = feature_v)

        self.model_c = extractor_c.to(device)
        self.intermediate_conv = InterMediateConv().to(device)
        self.model_v = extractor_v.to(device)        
        self.MLP = LastModule().to(device)
        print("ready");


    def forward(self, inputs):
        x = None
        for input in inputs:
            input = input.to(device)
            if x is None:
                x = input
            else:
                x = torch.concat([x,input],dim=1)
        x=self.model_c(x)
        x=self.intermediate_conv(x['out'])
        x=self.model_v(x)
        x=self.MLP(x['out'])
        return x
    
    def getAlgorithm(self):
        return self.algorithm


def create_model():
    model = Resnet34_VIT448_SE()
    return model
