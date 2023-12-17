import torch
import metric as f
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def classification(model, params):
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
    len_data = len(dataset_dl.sampler)
    incorrect_output = f.IncorrectOutput(columns_name=["1++","1+","1","2","3"])
    confusion_matrix = f.ConfusionMatrix()
    metrics = f.Metrics(eval_function, num_classes, 'regression', len_data, columns_name)

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
    for val_metric in val_metrics.getMetrics():
        print(f'Validation {val_metric.getClassName()} is {val_metric.getResult()}')
