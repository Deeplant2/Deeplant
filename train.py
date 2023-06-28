import torch
import mlflow
from tqdm import tqdm
import CM as cm
import numpy as np
import pandas as pd
import os
import sklearn
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def training(model, params):
    num_epochs=params['num_epochs']
    loss_func=params['loss_func']
    optimizer=params['optimizer']
    scheduler=params['lr_scheduler']
    log_epoch=params['log_epoch']
    train_dl=params['train_dl']
    val_dl=params['val_dl']
    fold=params['fold']
    sanity=params['sanity_check']
    signature=params['signature']

    train_loss, val_loss, train_metric, val_metric =[], [], [], []
    best_acc = 0
    for epoch in tqdm(range(num_epochs)):

        if epoch == num_epochs - 1:
            df = pd.DataFrame(columns=['file_name', 'output', 'score', 'prediction'])
        else:
            df = None

        #training
        model.train()
        loss, metric, _ = loss_epoch(model, loss_func, train_dl, epoch, fold+1, sanity, optimizer)
        mlflow.log_metric("train loss", loss, epoch)
        mlflow.log_metric("train accuracy", metric, epoch)
        train_loss.append(loss)
        train_metric.append(metric)

        #validation
        model.eval()
        with torch.no_grad():
            loss, metric, df = loss_epoch(model, loss_func, val_dl, epoch, fold+1, sanity, df=df)
        mlflow.log_metric("val loss", loss, epoch)
        mlflow.log_metric("val accuracy", metric, epoch)
        val_loss.append(loss)
        val_metric.append(metric)
        scheduler.step(val_loss[-1])

        if epoch % log_epoch == log_epoch-1:
            mlflow.pytorch.log_model(model, f'model_fold_{fold+1}_epoch_{epoch}', signature=signature)
            
        #saving best model
        if val_metric[-1]>best_acc:
            best_acc = val_metric[-1]
            mlflow.set_tag("best", f'best at epoch {epoch}')
            mlflow.pytorch.log_model(model, f"best", signature=signature)
        print('The Validation Loss is {} and the validation accuracy is {}'.format(val_loss[-1],val_metric[-1]))
        print('The Training Loss is {} and the training accuracy is {}'.format(train_loss[-1],train_metric[-1]))

    if not os.path.exists('temp'):
        os.mkdir('temp')
    df.to_csv('temp/incorrect_data.csv')
    mlflow.log_artifact('temp/incorrect_data.csv')
    return model, train_metric, val_metric, train_loss, val_loss


# calculate the loss per epochs
def loss_epoch(model, loss_func, dataset_dl, epoch, fold, sanity_check=False, opt=None, df=None):
    running_loss = 0.0
    running_metrics = 0.0
    len_data = len(dataset_dl.sampler)

    conf_label = []
    conf_pred = []

    for xb, yb, name_b in tqdm(dataset_dl):
        xb = xb.to(device)
        yb = yb.to(device)
        output = model(xb)
        loss_b = loss_func(output, yb)
        scores, pred_b = torch.max(output.data,1)
        metric_b = (pred_b == yb).sum().item()
        running_loss += loss_b.item()
        
        if opt is not None:
            opt.zero_grad()
            loss_b.backward()
            opt.step()
        
        if opt is None:
            # Confusion Matrix에 쓰일 data append하는 부분
            predictions_conv = pred_b.cpu().numpy()
            labels_conv = yb.cpu().numpy()
            conf_pred.append(predictions_conv)
            conf_label.append(labels_conv)

            if df is not None:
                # 틀린 이미지 csv로 저장하는 부분
                index = torch.nonzero((pred_b != yb)).squeeze().cpu().tolist()
                if not isinstance(index, list):
                    index = [index]  # index가 단일 값인 경우에 리스트로 변환하여 처리
                scores = scores.cpu().numpy()
                output = list(output.detach().cpu().numpy())
                name_b = list(name_b)
                for i in index:
                    new_row = pd.DataFrame({'file_name':name_b[i], 'output':output[i], 'score':scores[i], 'prediction':predictions_conv[i]})
                    df = pd.concat([df,new_row], ignore_index=True)


        if metric_b is not None:
            running_metrics += metric_b

        if sanity_check is True:
            break

    # Validation 일 때 Confusion Matrix 그리는 부분
    if opt is None:   
        new_pred = np.concatenate(conf_pred)
        new_label = np.concatenate(conf_label)
        con_mat=sklearn.metrics.confusion_matrix(new_label, new_pred)
        CM=cm.SaveCM(con_mat,epoch)
        CM.save_plot(fold)

    loss = running_loss / len_data
    metric = running_metrics / len_data
    return loss, metric, df

