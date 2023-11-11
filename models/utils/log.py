import mlflow
import numpy as np


# +
def logArrayMlflow(name, array):
    for i in range(len(array)):
        mlflow.log_metric(name, array[i], i)
        
        
def convertEvalName(name):
    if name == "R2S":
        return "r2score"
    elif name == "ACC":
        return 'accuracy'
    elif name == "FACC":
        return 'floor accuracy'
    elif name == 'RACC':
        return 'round accuracy'
    elif name == "MAE":
        return "mae"
    
        
def logFoldMean(train_loss, val_loss, train_metric, val_metric, eval_function, columns_name):
    train_loss = np.array(train_loss).mean(axis=0)
    val_loss = np.array(val_loss).mean(axis=0)
    logArrayMlflow("train loss", train_loss)
    logArrayMlflow("val loss", val_loss)
    
    # axis=3를 기준으로 나누기
    train_metric = [[[item[j][i] for j in range(len(item))] for i in range(len(item[0]))] for item in train_metric]

    for j in range(len(train_metric[0])):
        temp = []
        for i in range(len(train_metric)):
            temp.append(train_metric[i][j])
        temp = np.array(temp).mean(axis=0)
        if temp.ndim == 1:
            logArrayMlflow(f"train {convertEvalName(eval_function[j])}", temp)
        else:
            for i in range(len(columns_name)):
                logArrayMlflow(f"train {convertEvalName(eval_function[j])} {columns_name[i]}", temp[:,i])
                
    val_metric = [[[item[j][i] for j in range(len(item))] for i in range(len(item[0]))] for item in val_metric]

    for j in range(len(val_metric[0])):
        temp = []
        for i in range(len(val_metric)):
            temp.append(val_metric[i][j])
        temp = np.array(temp).mean(axis=0)
        if temp.ndim == 1:
            logArrayMlflow(f"val {convertEvalName(eval_function[j])}", temp)
        else:
            for i in range(len(columns_name)):
                logArrayMlflow(f"val {convertEvalName(eval_function[j])} {columns_name[i]}", temp[:,i])
    
    
    

# -


