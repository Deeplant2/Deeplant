import mlflow
import numpy as np


def logArrayMlflow(name, array):
    '''
    #1. 함수명: logArrayMlflow\n
    #2. 목적/용도: 입력으로 받은 array를 mlflow에 기록하는 함수.\n 
    #3. Input parameters:\n
    name = mlflow에 저장될 metric 이름.\n
    array = 1차원 배열로 mlflow에 저장될 데이터가 시간 순으로 적힌 배열.\n
    '''
    for i in range(len(array)):
        mlflow.log_metric(name, array[i], i)

def convertEvalName(name):
    '''
    #1. 함수명: convertEvalName\n
    #2. 목적/용도: eval_function에 적혀 있는 것을 더 직관적인 이름으로 변환하는 함수.\n 
    #3. Input parameters:\n
    name = eval_function에 적힌 이름.\n
    #4. Output: 변환된 이름.\n
    #5. 기타 참고사항: metric을 추가할 때마다 바꿔줘야 하는 불편함이 있다. eval_function의 이름을 그대로 쓸거면 삭제해도 문제없다.
    '''
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
    '''
    #1. 함수명: logFoldMean\n
    #2. 목적/용도: cross validation 진행 시 전체 fold의 평균은 기록이 안 되므로 평균을 기록하기 위해 만든 함수\n 
    #3. Input parameters:\n
    train_loss = 각 fold의 train_loss가 기록된 리스트\n
    val_loss = 각 fold의 val_loss가 기록된 리스트\n
    train_metric = 각 fold의 train_metric이 기록된 리스트\n
    val_metric = 각 fold의 val_metric이 기록된 리스트\n
    eval_function = 모델 학습 시 사용된 metric 리스트. configuration file에 정의되어 있다.\n
    columns_name = configuration file의 output_column의 이름.\n
    '''
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