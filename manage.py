import torch
import pandas as pd
import numpy as np

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torch import optim
import matplotlib.pyplot as plt

import timm
import mlflow
import os
import gc

from torch import nn
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
from torch.utils.data import random_split, SubsetRandomSampler
from torch.utils.data import DataLoader
from mlflow.models.signature import infer_signature
from sklearn.model_selection import KFold

import train
import argparse
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, TensorSpec
from datetime import datetime as dt

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)


def grade_encoding(x):
    if x == '1++':
        return 0
    elif x == '1+':
         return 1
    elif x == '1':
        return 2
    elif x == '2':
        return 3
    elif x == '3':
        return 4
    return 0

class CreateImageDataset(Dataset):
    def __init__(self, labels, img_dir, transform=None, target_transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.img_labels = labels

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        
        label = self.img_labels.iloc[idx]['grade_encode']
        grade = self.img_labels.iloc[idx]['grade']
        name = self.img_labels.iloc[idx]['file_name']
        img_folder = f'grade_{grade}'
        img_path = os.path.join(self.img_dir, img_folder)
        img_path = os.path.join(img_path, self.img_labels.iloc[idx, 0])
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label, name



parser=argparse.ArgumentParser(description='training pipeline for image classification')

parser.add_argument('--model_type', default ='vit', type=str)  # 사용할 모델 선택
parser.add_argument('--model_name', default='vit_base_patch32_clip_448.laion2b_ft_in12k_in1k', type=str)  # 사용할 세부 모델 선택
parser.add_argument('--run_name', default=None, type=str)  # run 이름 선택
parser.add_argument('--sanity', default=False, type=bool)  # 빠른 test 여부
parser.add_argument('--mode', default='train', type=str) # 학습모드 / 평가모드
parser.add_argument('--image_size', default=448, type=int, choices=(224,448))  # 이미지 크기 재설정
parser.add_argument('--num_workers', default=4, type=int)  # 훈련에 사용할 코어 수

parser.add_argument('--epochs', default=10, type=int)  # fold당 epoch
parser.add_argument('--kfold', default=5, type=int)  # kfold 사이즈
parser.add_argument('--batch_size', default=16, type=int)  # 배치 사이즈
parser.add_argument('--patch_size', default=None, type=int)  # 패치 사이즈
parser.add_argument('--lr', '--learning_rate', default=1e-5, type=float)  # learning rate
parser.add_argument('--log_epoch', default=10, type=int)  # 몇 epoch당 기록할 지 정함
parser.add_argument('--num_classes', default=5, type=int)  # output class 개수

parser.add_argument('--factor', default=0.5, type=float)  # scheduler factor
parser.add_argument('--threshold', default=0.003, type=float)  # scheduler threshold
parser.add_argument('--momentum', default=0.9, type=float)  # optimizer의 momentum
parser.add_argument('--weight_decay', '--wd', default=5e-4, type=float)  # 가중치 정규화


parser.add_argument('--data_path', default='/home/work/resized_image_datas/image_5class_5000/448/', type=str)  # data path

#parser.add_argument('--optim', default='ADAM')  # optimizer
parser.add_argument('--pretrained', default=True, type=bool, help='use pre-trained model')  # pre-train 모델 사용 여부
parser.add_argument('--load_run', default=False, type=bool, help='use runned model')  # run의 모델 사용 여부
parser.add_argument('--logged_model', default=None, type=str, help='logged model path') # 사용할 run의 path

args=parser.parse_args()



#Define data pathes
#homepath = '/home/work/resized_image_datas/image_5class_5000/'
image_size = args.image_size
datapath = args.data_path
trainpath = os.path.join(datapath,'Training')
valpath = os.path.join(datapath,'Valid')

train_label_set = pd.read_csv(f'{datapath}/train.csv')
val_label_set = pd.read_csv(f'{datapath}/valid.csv')

train_label_set['grade_encode'] = train_label_set['grade'].apply(grade_encoding)
val_label_set['grade_encode'] = val_label_set['grade'].apply(grade_encoding)

#Define kfold
kfold = args.kfold

#Define input transform
transformation = transforms.Compose([
transforms.RandomHorizontalFlip(p=0.3),
transforms.RandomVerticalFlip(p=0.3),
transforms.RandomRotation((-20,20)),
transforms.ToTensor(),
])

#Define Data loader
train_dataset = CreateImageDataset(train_label_set, trainpath, transform=transformation)
valid_dataset = CreateImageDataset(val_label_set, valpath, transform=transformation)
dataset = torch.utils.data.ConcatDataset([train_dataset, valid_dataset])
splits = KFold(n_splits = kfold, shuffle = True, random_state =42)

#Define hyperparameters
batch_size = args.batch_size
lr = args.lr
epochs = args.epochs

log_epoch = args.log_epoch
num_workers = args.num_workers
num_classes = args.num_classes

sanity = args.sanity
pretrained = args.pretrained
model_name = args.model_name

load_run = args.load_run
logged_model = args.logged_model #'runs:/523f68657d884879844be1c409bd96c0/best'

experiment_name = args.model_type
run_name = args.run_name



mlflow.set_experiment(experiment_name)
now = dt.now()
date_time_string = now.strftime("%Y-%m-%d %H:%M:%S")

train_acc_sum = np.zeros(epochs)
val_acc_sum = np.zeros(epochs)
train_loss_sum = np.zeros(epochs)
val_loss_sum = np.zeros(epochs)

if run_name == None:
    run_name = experiment_name
else:
    run_name = run_name

# Start running
with mlflow.start_run(run_name=run_name) as parent_run:
    print(parent_run.info.run_id)
    if load_run == False:
        mlflow.log_param("model_name", model_name)
        mlflow.log_param('pretrained', pretrained)
    else:
        mlflow.log_param("model_name", logged_model)
        mlflow.log_param("pretrained", True)

    mlflow.log_param("num_epochs", epochs)
    mlflow.log_param("learning_rate", lr)
    mlflow.log_param('batch_size', batch_size)
    mlflow.log_param("image_size", image_size)

    loss_func = nn.CrossEntropyLoss()
    mlflow.log_param("loss_func", loss_func)

    for fold, (train_idx,val_idx) in enumerate(splits.split(np.arange(len(dataset)))):            
        print('Fold {}'.format(fold + 1))

        if load_run == True:
            model = mlflow.pytorch.load_model(logged_model)
        else:
            model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)

        if args.model_type == 'vit' and args.patch_size is not None:
            model.patch_embed.patch_size = (args.patch_size, args.patch_size)
        
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr = lr)
        scheduler = ReduceLROnPlateau(optimizer, patience = 2, factor = args.factor, threshold = args.threshold)

        train_sampler = SubsetRandomSampler(train_idx) 
        test_sampler = SubsetRandomSampler(val_idx)
        train_dl = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers, pin_memory=True)
        val_dl = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler, num_workers=num_workers, pin_memory=True)

        input_schema = Schema([TensorSpec(np.dtype(np.float32),shape=(image_size,image_size))])
        output_schema = Schema([TensorSpec(np.dtype(np.float32), (1, num_classes))])
        signature = ModelSignature(inputs=input_schema, outputs=output_schema)

        # train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        # val_dl = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        params_train = {
        'num_epochs':epochs,
        'optimizer':optimizer,
        'loss_func':loss_func,
        'train_dl':train_dl,
        'val_dl':val_dl,
        'sanity_check':sanity,
        'lr_scheduler':scheduler,
        'log_epoch':log_epoch,
        'fold':fold,
        'signature':signature,
        }

        with mlflow.start_run(run_name=str(fold+1), nested=True) as run:
            model, train_acc, val_acc, train_loss, val_loss = train.training(model, params_train)
            train_acc_sum += train_acc
            val_acc_sum += val_acc
            train_loss_sum += train_loss
            val_loss_sum += val_loss

            #plot the curves
            plt.plot(train_acc, label = 'train_acc')
            plt.plot(val_acc, label = 'val_acc')
            plt.plot(train_loss, label = 'train_loss')
            plt.plot(val_loss, label = 'val_loss')
            plt.legend()
            plt.title('Accuracy and Loss Plots')
            figure = plt.gcf()
            mlflow.log_figure(figure, "Graph_"+str(log_epoch)+'_'+str(fold+1)+'.jpg')
            plt.clf()


        model.cpu()
        del model
        gc.collect()
    for i in range(epochs):
        mlflow.log_metric("train loss", train_loss_sum[i] / kfold, i)
        mlflow.log_metric("train accuracy", train_acc_sum[i] / kfold , i)
        mlflow.log_metric("val loss", val_loss_sum[i] / kfold, i)
        mlflow.log_metric("val accuracy", val_acc_sum[i] / kfold, i)

torch.cuda.empty_cache()
