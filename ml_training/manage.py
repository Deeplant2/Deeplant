import os
import gc
import torch
import mlflow
import numpy as np
import pandas as pd

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from torch import optim
from torch import nn

import train
import test
import models.make_model as m
import utils.dataset as dataset
import utils.loss as loss
import utils.log as log
import utils.data_dist as data_dist
import argparse
import json

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
#-----------------------------------------------------------------------------------------------------------------

parser=argparse.ArgumentParser(description='training pipeline for image classification')

parser.add_argument('--run', default ='proto', type=str)  # run 이름 설정
parser.add_argument('--ex', '--experiment', default ='proto', type=str)  # experiment 이름 설정
parser.add_argument('--model_cfgs', type=str)  # model config 파일 경로 설정
parser.add_argument('--mode', default='train', type=str, choices=('train', 'test')) # 학습모드 / 평가모드
parser.add_argument('--epochs', default=10, type=int)  #epochs
parser.add_argument('--log_epoch', default=10, type=int)  # save model per log epochs
parser.add_argument('--lr', '--learning_rate', default=1e-5, type=float)  # learning rate
parser.add_argument('--data_path', default='/home/work/deeplant_data', type=str)  # data path
parser.add_argument('--sanity', default=False, type=bool) # test mode
parser.add_argument('--csv_name', default='new_train.csv')
args=parser.parse_args()

# -----------------------------------------------------------------------------------------------------------------
# Read Model's configs
with open(args.model_cfgs, 'r') as json_file:
    model_cfgs = json.load(json_file)

#Define hyperparameters
params = model_cfgs['hyperparameters']

num_workers = params['num_workers']
batch_size = params['batch_size']
factor = params['factor']
threshold = params['threshold']
momentum = params['momentum']
weight_decay = params['weight_decay']
seed = params['seed']
save_model = params['save_model']
eval_function = params['eval_function']
cross_validation = params['cross_validation']

epochs = args.epochs
lr = args.lr
experiment_name = args.ex
run_name = args.run
log_epoch = args.log_epoch
sanity = args.sanity
csv_name = args.csv_name
mode = args.mode

#Define data pathes
datapath = args.data_path
label_set = pd.read_csv(os.path.join(datapath,csv_name))

output_columns = model_cfgs['output_columns']
columns_name = label_set.columns[output_columns].values
print(columns_name)

# Define dataset
# cross validation 사용 시 dataset은 아래에서 선언함.
if mode == 'train':
    if cross_validation == 0:
        train_set, test_set = train_test_split(label_set, test_size=0.1, random_state= seed)
        train_set.reset_index(inplace=True, drop=True)
        test_set.reset_index(inplace=True, drop=True)
        print(train_set)
        print(test_set)

        train_dataset = dataset.CreateImageDataset(train_set, datapath, model_cfgs['datasets'], output_columns)
        test_dataset = dataset.CreateImageDataset(test_set, datapath, model_cfgs['datasets'], output_columns)
    else:
        splits = KFold(n_splits = cross_validation, shuffle = True, random_state = seed)
        
elif mode == 'test':
    test_dataset = dataset.CreateImageDataset(label_set, datapath, model_cfgs['datasets'], output_columns)
# ------------------------------------------------------

# mlflow 설정
mlflow.set_tracking_uri('file:///home/work/model/multi_input_model/mlruns/')
mlflow.set_experiment(experiment_name)

# mlflow를 시작 
with mlflow.start_run(run_name=run_name) as parent_run:
    print(parent_run.info.run_id)
    mlflow.log_dict(model_cfgs, 'config/configs.json')
    mlflow.log_param("num_epochs", epochs)
    mlflow.log_param("learning_rate", lr)
    mlflow.log_param("log_epoch", log_epoch)

    params_train = {
        'num_epochs':epochs,
        'optimizer':None,
        'train_dl':None,
        'val_dl':None,
        'lr_scheduler':None,
        'log_epoch':log_epoch,
        'num_classes':model_cfgs['models']['num_classes'],
        'columns_name':columns_name,
        'eval_function':eval_function,
        'save_model':save_model,
        'loss_func':None,
        'sanity':sanity
    }
        
    if args.mode =='train':
        if cross_validation == 0:
            model = m.create_model(model_cfgs)
            model = model.to(device)
            
            algorithm = model.getAlgorithm()
            
            total_params = sum(p.numel() for p in model.parameters())
            mlflow.log_param("total_parmas", total_params)
            
            data_dist.logDatasetHistogram(train_set, test_set, columns_name)
            
            params_train['optimizer'] = optim.Adam(model.parameters(), lr = lr)
            params_train['lr_scheduler'] = ReduceLROnPlateau(params_train['optimizer'], patience = 2, factor = factor, threshold = threshold)
            params_train['train_dl'] = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
            params_train['val_dl'] = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
            
            
            if algorithm == 'classification':
                params_train['columns_name'] = label_set[output_columns].unique()
                params_train['loss_func'] = nn.CrossEntropyLoss()
                mlflow.log_param("loss_func", params_train['loss_func'])
                model, _, _, _, _ = train.classification(model, params_train)
            elif algorithm == 'regression':
                #params_train['loss_func'] = loss.DenseWeightMSELoss(alpha=1, y=np.array(label_set[columns_name]))
                params_train['loss_func'] = nn.MSELoss()
                mlflow.log_param("loss_func", params_train['loss_func'])
                model, _, _, _, _ = train.regression(model, params_train)
                
 
            model.cpu()
            del model
            gc.collect()

        else:
            train_loss_list, val_loss_list, train_metric_list, val_metric_list = [],[],[],[]
            for fold, (train_idx,val_idx) in enumerate(splits.split(np.arange(len(label_set)))):
                model = m.create_model(model_cfgs)
                model = model.to(device)
                
                total_params = sum(p.numel() for p in model.parameters())
                mlflow.log_param("total_parmas", total_params)
                
                params_train['optimizer'] = optim.Adam(model.parameters(), lr = lr)
                params_train['lr_scheduler'] = ReduceLROnPlateau(params_train['optimizer'], patience = 2, factor = factor, threshold = threshold)
                train_dataset = dataset.CreateImageDataset(label_set.iloc[train_idx], datapath, model_cfgs['datasets'], output_columns, train=True)
                val_dataset = dataset.CreateImageDataset(label_set.iloc[val_idx], datapath, model_cfgs['datasets'], output_columns, train=False)
                params_train['train_dl'] = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
                params_train['val_dl'] = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
                print(label_set.iloc[train_idx])
                print(label_set.iloc[val_idx])
                
                algorithm = model.getAlgorithm()
                with mlflow.start_run(run_name=str(fold+1), nested=True) as run:
                    print(f"Fold {fold+1}: {run.info.run_id}")
                    data_dist.logDatasetHistogram(label_set.iloc[train_idx], label_set.iloc[val_idx], columns_name)
                    if algorithm == 'classification':
                        params_train['columns_name'] = label_set[columns_name[0]].unique()
                        params_train['loss_func'] = nn.CrossEntropyLoss()
                        mlflow.log_param("loss_func", params_train['loss_func'])
                        model, train_loss, val_loss, train_metric, val_metric = train.classification(model, params_train)
                    elif algorithm == 'regression':
                        #params_train['loss_func'] = loss.DenseWeightMSELoss(alpha=0.3, y=np.array(label_set[columns_name]))
                        params_train['loss_func'] = nn.MSELoss()
                        mlflow.log_param("loss_func", params_train['loss_func'])
                        model, train_loss, val_loss, train_metric, val_metric = train.regression(model, params_train)
                        
                train_loss_list.append(train_loss)
                val_loss_list.append(val_loss)
                train_metric_list.append(train_metric)
                val_metric_list .append(val_metric)
                model.cpu()
                del model
                gc.collect()
            log.logFoldMean(train_loss_list, val_loss_list, train_metric_list, val_metric_list, eval_function, columns_name)
            
            
    elif args.mode =='test':
        model = m.create_model(model_cfgs)
        model = model.to(device)

        algorithm = model.getAlgorithm()
        params_train['val_dl'] = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

        if algorithm == 'classification':
            params_train['columns_name'] = label_set[output_columns].unique()
            model = test.classification(model, params_train)
        elif algorithm == 'regression':
            model = test.regression(model, params_train)

        model.cpu()
        del model
        gc.collect()

torch.cuda.empty_cache()
