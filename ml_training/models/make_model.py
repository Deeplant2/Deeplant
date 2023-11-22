import torch
import timm
import importlib

def create_model(model_cfgs):

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
