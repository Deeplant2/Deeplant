# +
from torchvision import transforms


def create_transform(dataset_cfgs):
    
    train_transforms, val_transforms = [], []
    
    for dataset_cfg in dataset_cfgs:
        train_transform = dataset_cfg['train_transform']
        train = []
        if train_transform['Resize'] is not None:
            train.append(transforms.Resize(train_transform['Resize']))
        if train_transform['RandomHorizontalFlip'] is not None:
            train.append(transforms.RandomHorizontalFlip(p=train_transform['RandomHorizontalFlip']))
        if train_transform['RandomVerticalFlip'] is not None:
            train.append(transforms.RandomVerticalFlip(p=train_transform['RandomVerticalFlip']))
        if train_transform['RandomRotation'] is not None:
            train.append(transforms.RandomRotation(train_transform['RandomRotation']))
        if train_transform['Grayscale'] is not None:
            train.append(transforms.Grayscale(num_output_channels=train_transform['Grayscale']))
        if train_transform['ToTensor'] is True:
            train.append(transforms.ToTensor())
            
        if train: 
            train_transforms.append(transforms.Compose(train))
        else:
            train_transforms.append(None)
        
        val_transform = dataset_cfg['val_transform']
        val = []
        if val_transform['Resize'] is not None:
            val.append(transforms.Resize(val_transform['Resize']))
        if val_transform['RandomHorizontalFlip'] is not None:
            val.append(transforms.RandomHorizontalFlip(p=val_transform['RandomHorizontalFlip']))
        if val_transform['RandomVerticalFlip'] is not None:
            val.append(transforms.RandomVerticalFlip(p=val_transform['RandomVerticalFlip']))
        if val_transform['RandomRotation'] is not None:
            val.append(transforms.RandomRotation(val_transform['RandomRotation']))
        if val_transform['Grayscale'] is not None:
            val.append(transforms.Grayscale(num_output_channels=val_transform['Grayscale']))
        if val_transform['ToTensor'] is True:
            val.append(transforms.ToTensor())
            
        if val:
            val_transforms.append(transforms.Compose(val))
        else:
            val_transforms.append(None)
    
    return train_transforms, val_transforms



