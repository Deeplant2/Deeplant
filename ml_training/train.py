import torch
import mlflow
import metric as f
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def classification(model, params):
    '''
    #1. 함수명: classification\n
    #2. 목적/용도: classification을 수행하는 함수\n 
    #3. Input parameters:\n
    model: 학습에 사용될 모델\n 
    params: 학습 파라미터가 담긴 dictionary\n
    #4. Output: 모델 및 모든 epoch의 결과가 담긴 list\n
    '''
    num_epochs=params['num_epochs']
    loss_func=params['loss_func']
    optimizer=params['optimizer']
    scheduler=params['lr_scheduler']
    log_epoch=params['log_epoch']
    train_dl=params['train_dl']
    val_dl=params['val_dl']
    columns_name=params['columns_name']
    eval_function=params['eval_function']
    save_model=params['save_model']
    sanity=params['sanity']

    best_loss = -1
    train_loss_list, val_loss_list, train_metric_list, val_metric_list =[], [], [], []
    for epoch in tqdm(range(num_epochs)):

        #training
        model.train()
        train_loss, train_metrics = classification_epoch(model, loss_func, train_dl, epoch, eval_function, 1, columns_name, sanity, optimizer)
        train_loss_list.append(train_loss)
        train_metric_list.append(train_metrics.getResults())
        #validation
        model.eval()
        with torch.no_grad():
            val_loss, val_metrics= classification_epoch(model, loss_func, val_dl, epoch, eval_function, 1, columns_name, sanity)
        scheduler.step(val_loss)
        val_loss_list.append(val_loss)
        val_metric_list.append(val_metrics.getResults())


        mlflow.log_metric("train loss", train_loss, epoch)
        mlflow.log_metric("val loss", val_loss, epoch)
        train_metrics.logMetrics("train", epoch)
        val_metrics.logMetrics("val", epoch)
        printResults(train_loss, train_metrics, val_loss, val_metrics)
        
        if save_model is True:
            best_loss = saveModel(model, epoch, log_epoch, val_loss, best_loss)

    return model, train_loss_list, val_loss_list, train_metric_list, val_metric_list


# calculate the loss per epochs
def classification_epoch(model, loss_func, dataset_dl, epoch, eval_function, num_classes, columns_name, sanity=False, opt=None):
    '''
    #1. 함수명: classification_epoch\n
    #2. 목적/용도: classification의 한 반복을 수행하는 함수\n 
    #3. Input parameters:\n
    model: 테스트에 사용될 모델\n
    loss_func: loss 함수\n
    dataset_dl: 데이터 로더\n
    epoch: 현재 반복 횟수\n
    eval_function: 수치적으로 평가할 metric 이름\n
    num_classes: 예측열의 개수\n
    columns_name: 예측열의 이름.\n
    sanity: 코드 테스트를 위한 flag. true면 1배치만 실행하고 나머지는 스킵함.\n
    opt: 학습 최적화에 사용되는 객체. validation일 땐 안 쓰임.\n.
    #4. Output: loss 및 metric 결과\n
    '''
    running_loss = 0.0
    len_data = len(dataset_dl.sampler)

    incorrect_output = f.IncorrectOutput(class_name=["1++","1+","1","2","3"])
    confusion_matrix = f.ConfusionMatrix()
    metrics = f.Metrics(eval_function, num_classes, 'classification', len_data, columns_name)

    for xb, yb, name_b in tqdm(dataset_dl):
        yb = yb.to(device).long()
        yb = yb[:,0]
        output = model(xb)
        loss_b = loss_func(output, yb)

        metrics.update(output, yb)
    
        # L1 regularization =0.001
        lambda1= 0.0000003
        l1_regularization =0.0
        for param in model.parameters():
            l1_regularization += torch.norm(param,1)
        l1_regularization = lambda1 * l1_regularization
        
        running_loss += loss_b.item() + l1_regularization.item()
        loss_b = loss_b + l1_regularization

        if opt is not None:
            opt.zero_grad()
            loss_b.backward()
            opt.step()
    
        # Validation
        if opt is None:
            confusion_matrix.update(output, yb)
            incorrect_output.update(output, yb, name_b)

        if sanity is True:
            break

    # Validation
    if opt is None: 
        confusion_matrix.logMetric(epoch=epoch)
        incorrect_output.logMetric(filename="incorrect_output.csv", epoch=epoch)

    loss = running_loss / len(dataset_dl)
    return loss, metrics


def regression(model, params):
    '''
    #1. 함수명: regression\n
    #2. 목적/용도: regression을 수행하는 함수\n 
    #3. Input parameters:\n
    model: 학습에 사용될 모델\n 
    params: 학습 파라미터가 담긴 dictionary\n
    #4. Output: 모델 및 모든 epoch의 결과가 담긴 list\n
    '''
    num_epochs=params['num_epochs']
    loss_func=params['loss_func']
    optimizer=params['optimizer']
    scheduler=params['lr_scheduler']
    log_epoch=params['log_epoch']
    train_dl=params['train_dl']
    val_dl=params['val_dl']
    num_classes=params['num_classes']
    columns_name=params['columns_name']
    eval_function=params['eval_function']
    save_model=params['save_model']
    sanity=params['sanity']
    
    train_loss_list, val_loss_list, train_metric_list, val_metric_list =[], [], [], []
    best_loss = -1.0
    for epoch in tqdm(range(num_epochs)):
        
        #training
        model.train()
        train_loss, train_metrics = regression_epoch(model, loss_func, train_dl, epoch, num_classes, columns_name, eval_function, sanity, optimizer)
        train_loss_list.append(train_loss)
        train_metric_list.append(train_metrics.getResults())
        
        #validation
        model.eval()
        with torch.no_grad():
            val_loss, val_metrics = regression_epoch(model, loss_func, val_dl, epoch, num_classes, columns_name, eval_function, sanity)
        scheduler.step(val_loss)
        val_loss_list.append(val_loss)
        val_metric_list.append(val_metrics.getResults())
        
        mlflow.log_metric("train loss", train_loss, epoch)
        mlflow.log_metric("val loss", val_loss, epoch)
        train_metrics.logMetrics("train", epoch)
        val_metrics.logMetrics("val", epoch)
        printResults(train_loss, train_metrics, val_loss, val_metrics)
        
        if save_model is True:
            best_loss = saveModel(model, epoch, log_epoch, val_loss, best_loss)

    return model, train_loss_list, val_loss_list, train_metric_list, val_metric_list


# calculate the loss per epochs
def regression_epoch(model, loss_func, dataset_dl, epoch, num_classes, columns_name, eval_function, sanity=False, opt=None):
    '''
    #1. 함수명: regression_epoch\n
    #2. 목적/용도: regression의 한 반복을 수행하는 함수\n 
    #3. Input parameters:\n
    model: 테스트에 사용될 모델\n
    loss_func: loss 함수\n
    dataset_dl: 데이터 로더\n
    epoch: 현재 반복 횟수\n
    eval_function: 수치적으로 평가할 metric 이름\n
    num_classes: 예측열의 개수\n
    columns_name: 예측열의 이름.\n
    sanity: 코드 테스트를 위한 flag. true면 1배치만 실행하고 나머지는 스킵함.\n
    opt: 학습 최적화에 사용되는 객체. validation일 땐 안 쓰임.\n.
    #4. Output: loss 및 metric 결과\n
    '''
    running_loss = 0.0
    len_data = len(dataset_dl.sampler)
    metrics = f.Metrics(eval_function, num_classes, 'regression', len_data, columns_name)
    output_log = f.OutputLog(columns_name, num_classes)

    for xb, yb, name_b in tqdm(dataset_dl):
        yb = yb.to(device)
        output = model(xb)
        
        loss = loss_func(output, yb)
        running_loss += loss.item()
        
        output_log.update(output, yb, name_b)
        metrics.update(output, yb)

        if opt is not None:
            opt.zero_grad()
            loss.backward()
            opt.step()     
            
        if sanity is True:
            break
        
    output_log.logMetric(epoch, opt)
    loss = running_loss / len(dataset_dl)
    return loss, metrics


def printResults(train_loss, train_metrics, val_loss, val_metrics):
    '''
    #1. 함수명: printResults\n
    #2. 목적/용도: loss 및 metric결과를 콘솔창에 출력하는 함수\n 
    #3. Input parameters:\n
    train_loss: train loss 결과\n
    train_metrics: train metric 결과\n
    val_loss: valid loss 결과\n
    val_metrics: valid metric 결과\n
    '''
    print('The Training Loss is {} and the Validation Loss is {}'.format(train_loss, val_loss))
    for train_metric, val_metric in zip(train_metrics.getMetrics(), val_metrics.getMetrics()):
        print(f'The Training {train_metric.getClassName()} is {train_metric.getResult()} and the Validation {val_metric.getClassName()} is {val_metric.getResult()}')


def saveModel(model, epoch, log_epoch, val_loss, best_loss):
    '''
    #1. 함수명: saveModel\n
    #2. 목적/용도: 모델을 mlflow에 저장하는 함수\n 
    #3. Input parameters:\n
    model: 학습에 사용중인 모델\n
    epoch: 현재 반복 횟수\n
    log_epoch: 몇 epoch마다 모델을 저장할지 정하는 주기\n
    val_loss: valid loss 결과\n
    best_loss: 지금까지의 loss 중 가장 낮은 loss\n
    #4. Output: best loss.\n
    '''
    if epoch % log_epoch == log_epoch-1:
        mlflow.pytorch.log_model(model, f'model_epoch_{epoch}')
    #saving best model
    if val_loss<best_loss or best_loss<0.0:
        best_loss = val_loss
        mlflow.set_tag("best", f'best at epoch {epoch}')
        mlflow.pytorch.log_model(model, f"best")
    return best_loss
