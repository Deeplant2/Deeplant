import torch
import os
from PIL import Image
from torch.utils.data import Dataset
import utils.transform as transform

class CreateImageDataset(Dataset):
    def __init__(self, labels, img_dir, dataset_cfgs, output_columns, train=True):

        self.train_transforms, self.val_transforms = transform.create_transform(dataset_cfgs)
        self.image_sizes = []
        self.isImage = []
        self.input_columns = []
        self.output_columns = output_columns
        self.model_cnt = len(dataset_cfgs)

        for dataset_cfg in dataset_cfgs:
            image_size = dataset_cfg['image_size']
            isImage = dataset_cfg['isImage']
            input_column = dataset_cfg['input_column']
            self.image_sizes.append(image_size)
            self.isImage.append(isImage)
            self.input_columns.append(input_column)

        self.img_dir = img_dir
        self.img_labels = labels
        self.train = train
        
        
    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        inputs = []
        outputs = torch.tensor(self.img_labels.iloc[idx, self.output_columns], dtype=torch.float32)
        for i in range(self.model_cnt):
            if self.isImage[i] == True:
                name = self.img_labels.iloc[idx, self.input_columns[i]]
                img_path = os.path.join(self.img_dir, name)
                image = Image.open(img_path)

                if self.train:
                    image = self.train_transforms[i](image)
                else:
                    image = self.val_transforms[i](image)
                inputs.append(image)
            else:
                input = torch.tensor(self.img_labels.iloc[idx, self.input_columns[i]], dtype=torch.float32)
                inputs.append(input) 

        return inputs, outputs, name
    
    