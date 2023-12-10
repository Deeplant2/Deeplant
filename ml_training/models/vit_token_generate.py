import torch
import torch.nn.functional as F
from torch import nn
import timm
from torchvision.models.feature_extraction import create_feature_extractor
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 사전 학습된 vit의 latent vector를 다른 vit의 token으로 embedding하는 모델.
# 현재 작성된 코드는 UTKFace 데이터를 가지고 gender를 학습한 모델을 불러와 latent vector를 뽑은 후 
# age를 예측하는 vit의 token으로 넣은 모델이다. gender 정보를 통해 age 예측 정확도를 높이기 위해 제작되었다.

class TokenModel(nn.Module):
    def __init__(self, model_name, num_classes):
        super(TokenModel,self).__init__()
        self.algorithm = "regression"
        
        model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
        pretrained_model = torch.load('models/pretrained_model/UTKFace_gender_2000_vit224.pth')
        self.token_model = create_feature_extractor(pretrained_model,{'model.fc_norm':'out'})
        
        for para in self.token_model.parameters():
            para.requires_grad = False
        
        module_list = list(model.children())
        self.cls_token = model.cls_token
        self.pos_embed = torch.concat([model.pos_embed[:,:1],model.pos_embed[:,:1],model.pos_embed[:,1:]],dim=1).to(device)
             
        self.patch_embed = torch.nn.Sequential(*module_list[0:4])
        self.encoder = module_list[4]
        self.fc_norm = module_list[5]
        self.head = model.head
        
    def _pos_embed(self, x):
        x = torch.cat((self.race_token.unsqueeze(dim=1), x), dim=1)
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + self.pos_embed
        return x
        
    def _head(self, x):
        x = x[:,0]
        x = self.fc_norm(x)
        x = self.head(x)
        return x
    
    def forward(self, inputs):
        x = inputs[0].to(device)
        t = self.token_model(x)['out']
        
        self.race_token = t
        
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.encoder(x)
        x = self._head(x)
    
        return x
    def getAlgorithm(self):
        return self.algorithm


def create_model(model_name, num_classes, in_chans, pretrained):
    model = TokenModel(model_name, num_classes)
    return model
