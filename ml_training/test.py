import torch
import metric as f
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def classification(model, params):
    '''
    #1. 함수명: classification\n
    #2. 목적/용도: classification을 수행하는 함수\n 
    #3. Input parameters:\n
    model: 테스트에 사용될 모델\n 
    params: 테스트 파라미터가 담긴 dictionary\n
    #4. Output: 모델\n
    '''
    num_epochs=params['num_epochs']
    val_dl=params['val_dl']
    columns_name=params['columns_name']
    eval_function=params['eval_function']
    sanity=params['sanity']

    for epoch in tqdm(range(num_epochs)):
        #validation
        model.eval()
        with torch.no_grad():
            val_metrics= classification_epoch(model, val_dl, epoch, eval_function, 1, columns_name, sanity)
        val_metrics.logMetrics("val", epoch)
        printResults(val_metrics)

    return model


# calculate the loss per epochs
def classification_epoch(model, dataset_dl, epoch, eval_function, num_classes, columns_name, sanity=False):
    '''
    #1. 함수명: classification_epoch\n
    #2. 목적/용도: classification의 한 반복을 수행하는 함수\n 
    #3. Input parameters:\n
    model: 테스트에 사용될 모델\n
    dataset_dl: 데이터 로더\n
    epoch: 현재 반복 횟수\n
    eval_function: 수치적으로 평가할 metric 이름\n
    num_classes: 예측열의 개수\n
    columns_name: 예측열의 이름.\n
    sanity: 코드 테스트를 위한 flag. true면 1배치만 실행하고 나머지는 스킵함.\n
    #4. Output: metric 결과\n
    '''
    len_data = len(dataset_dl.sampler)
    incorrect_output = f.IncorrectOutput(class_name=["1++","1+","1","2","3"])
    confusion_matrix = f.ConfusionMatrix()
    metrics = f.Metrics(eval_function, num_classes, 'classification', len_data, columns_name)

    for xb, yb, name_b in tqdm(dataset_dl):
        yb = yb.to(device).long()
        yb = yb[:,0]
        output = model(xb)
        metrics.update(output, yb)
        confusion_matrix.update(output, yb)
        incorrect_output.update(output, yb, name_b)

        if sanity is True:
            break

    confusion_matrix.logMetric(epoch=epoch)
    incorrect_output.logMetric(filename="incorrect_output.csv", epoch=epoch)

    return metrics


def regression(model, params):
    '''
    #1. 함수명: regression\n
    #2. 목적/용도: regression을 수행하는 함수\n 
    #3. Input parameters:\n
    model: 테스트에 사용될 모델\n 
    params: 테스트 파라미터가 담긴 dictionary\n
    #4. Output: 모델\n
    '''
    num_epochs=params['num_epochs']
    val_dl=params['val_dl']
    num_classes=params['num_classes']
    columns_name=params['columns_name']
    eval_function=params['eval_function']
    sanity=params['sanity']
    
    for epoch in tqdm(range(num_epochs)):
        #validation
        model.eval()
        with torch.no_grad():
            val_metrics = regression_epoch(model, val_dl, epoch, num_classes, columns_name, eval_function, sanity)
        val_metrics.logMetrics("val", epoch)
        printResults(val_metrics)

    return model


# calculate the loss per epochs
def regression_epoch(model, dataset_dl, epoch, num_classes, columns_name, eval_function, sanity=False, opt=None):
    '''
    #1. 함수명: regression_epoch\n
    #2. 목적/용도: regression의 한 반복을 수행하는 함수\n 
    #3. Input parameters:\n
    model: 테스트에 사용될 모델\n
    dataset_dl: 데이터 로더\n
    epoch: 현재 반복 횟수\n
    eval_function: 수치적으로 평가할 metric 이름\n
    num_classes: 예측 열의 개수\n
    columns_name: 예측 열의 이름.\n
    sanity: 코드 테스트를 위한 flag. true면 1배치만 실행하고 나머지는 스킵함.\n
    opt: 학습 최적화에 사용되는 객체. 테스트라서 안쓰임\n.
    #4. Output: metric 결과\n
    '''
    len_data = len(dataset_dl.sampler)
    metrics = f.Metrics(eval_function, num_classes, 'regression', len_data, columns_name)
    output_log = f.OutputLog(columns_name, num_classes)

    for xb, yb, name_b in tqdm(dataset_dl):
        yb = yb.to(device)
        output = model(xb)
        output_log.update(output, yb, name_b)
        metrics.update(output, yb)
            
        if sanity is True:
            break
        
    output_log.logMetric(epoch, opt)
    return metrics


def printResults(val_metrics):
    '''
    #1. 함수명: printResults\n
    #2. 목적/용도: metric결과를 콘솔창에 출력하는 함수\n 
    #3. Input parameters:\n
    val_metrics: metric 결과\n
    '''
    for val_metric in val_metrics.getMetrics():
        print(f'Validation {val_metric.getClassName()} is {val_metric.getResult()}')
