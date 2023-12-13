import torch
import importlib

def create_model(model_cfgs):
    '''
    #1. 함수명: create_model \n
    #2. 목적/용도: 모델 코드를 import해서 모델을 생성 후 manage.py에 모델을 return 하는 함수.\n 
    #3. Input parameters: 
    model_cfgs (dict type) = configuration file에서 models 부분을 입력으로 받는다.\n 
    #4. Output : 모델 객체를 반환한다.\n
    #5. 기타 참고사항\n
    '''
    model_cfg = model_cfgs['models']

    module = model_cfg['module']
    islogged = model_cfg['islogged']
    model_name = model_cfg["model_name"]
    pretrained = model_cfg["pretrained"]
    num_classes = model_cfg["num_classes"]
    in_chans = model_cfg['in_chans']
    
    if not islogged:
        temp_module = importlib.import_module(module)
        temp_model = temp_module.create_model(model_name, num_classes, in_chans, pretrained)
    else:
        temp_model = torch.load(model_name)

    return temp_model                  
