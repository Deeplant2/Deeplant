import pandas as pd
import numpy as np
import sklearn
import torch
import os
import mlflow
import seaborn as sns
import matplotlib.pyplot as plt

class Accuracy():
    def __init__(self, length, num_classes, algorithm, columns_name, method = None):
        self.num_classes = num_classes
        self.length = length
        self.algorithm = algorithm
        self.columns_name = columns_name
        self.method = method
        self.cumulative_metric = np.zeros(num_classes)

    def update(self, output, yb):
        if self.algorithm == 'classification':
            _, pred_b = torch.max(output.data,1)
            metric_b = (pred_b == yb).sum().item()
            self.cumulative_metric += metric_b

        elif self.algorithm == 'regression':
            if self.method == None:
                if self.num_classes != 1:
                    for i in range(self.num_classes):
                        self.cumulative_metric[i] += (torch.round(output[:,i]) == yb[:,i]).sum().item()
                else:
                    self.cumulative_metric += (torch.round(output) == yb).sum().item()
            elif self.method == "floor":
                if self.num_classes != 1:
                    for i in range(self.num_classes):
                        self.cumulative_metric[i] += (torch.floor(output[:,i]/10)*10 == torch.floor(yb[:,i]/10)*10).sum().item()
                else:
                    self.cumulative_metric += (torch.floor(output/10)*10 == torch.floor(yb/10)*10).sum().item()
            elif self.method == "round":
                if self.num_classes != 1:
                    for i in range(self.num_classes):
                        self.cumulative_metric[i] += (torch.round(output[:,i]/10)*10 == torch.round(yb[:,i]/10)*10).sum().item()
                else:
                    self.cumulative_metric += (torch.round(output/10)*10 == torch.round(yb/10)*10).sum().item()
        
    def getResult(self):
        return self.cumulative_metric / self.length
    
    def getDictResult(self):
        result = getResult()
        dict_result = {}
        for i in range(self.num_classes):
            result[self.columns_name[i]] = result[i]
        return dict_result
    
    def getClassName(self):
        return "accuracy"
    
    def logMetric(self, mode, epoch):  
        result = self.getResult()
        for i in range(self.num_classes):
            if self.method is None:
                mlflow.log_metric(f"{mode} accuracy {self.columns_name[i]}", result[i], epoch)
            else:
                mlflow.log_metric(f"{mode} {self.method} accuracy {self.columns_name[i]}", result[i], epoch)



class R2score():
    def __init__(self, length):
        self.length = length
        self.cumulative_y = None
        self.cumulative_output = None
    
    def update(self, output, yb):
        output = output.detach().cpu().numpy()
        yb = yb.cpu().numpy()
        if self.cumulative_y is None:
            self.cumulative_y = np.array(yb)
        else:
            self.cumulative_y = np.vstack((self.cumulative_y,yb))
            
        if self.cumulative_output is None:
            self.cumulative_output = np.array(output)
        else:
            self.cumulative_output = np.vstack((self.cumulative_output,output))

    def getResult(self):
        y_mean = self.cumulative_y.mean(axis=0)
        ssr = np.square(self.cumulative_y - self.cumulative_output).sum(axis=0)
        sst = np.square(self.cumulative_y - y_mean).sum(axis=0)
        r2_score = (1 - (ssr / sst)).mean()
        return r2_score
    
    def getDictResult(self):
        return self.getResult()
    
    def getClassName(self):
        return "r2score"
    
    def logMetric(self, mode, epoch):  
        mlflow.log_metric(f"{mode} r2score", self.getResult(), epoch)


class MeanAbsError():
    def __init__(self, length, num_classes, columns_name):
        self.num_classes = num_classes
        self.cumulative_metric = np.zeros(num_classes)
        self.length = length
        self.columns_name = columns_name
    
    def update(self, output, yb):
        if self.num_classes != 1:
            for i in range(self.num_classes):
                self.cumulative_metric[i] += torch.abs(output[:,i] - yb[:,i]).sum().item()
        else:
            self.cumulative_metric += torch.abs(output - yb).sum().item()

    def getResult(self):
        return self.cumulative_metric / self.length
    
    def getDictResult(self):
        result = self.getResult()
        dict_result = {}
        for i in range(self.num_classes):
            result[self.columns_name[i]] = result[i]
        return dict_result
    
    def getClassName(self):
        return "mae"
    
    def logMetric(self, mode, epoch):
        result = self.cumulative_metric / self.length
        for i in range(self.num_classes):
            mlflow.log_metric(f"{mode} mae {self.columns_name[i]}", result[i], epoch)


class Metrics():
    def __init__(self,eval_function, num_classes, algorithm, data_length, columns_name):
        self.metrics = []
        for f in eval_function:
            if f == 'ACC':
                self.metrics.append(Accuracy(data_length, num_classes, algorithm, columns_name))
            elif f == 'R2S':
                self.metrics.append(R2score(data_length))
            elif f == 'MAE':
                self.metrics.append(MeanAbsError(data_length, num_classes, columns_name))
            elif f == 'FACC':
                self.metrics.append(Accuracy(data_length, num_classes, algorithm, columns_name, 'floor'))
            elif f == 'RACC':
                self.metrics.append(Accuracy(data_length, num_classes, algorithm, columns_name, 'round'))

    def update(self, output, yb):
        for metric in self.metrics:
            metric.update(output, yb)
        
    def getMetrics(self):
        return self.metrics
    
    def logMetrics(self, mode, epoch):
        for metric in self.metrics:
            metric.logMetric(mode, epoch)
            
    def getResults(self):
        result = []
        for metric in self.metrics:
            result.append(metric.getResult())
        return result 
    
    def getDictResults(self):
        result = {}
        for i in range(len(self.metrics)):
            result[self.metrics[i].getClassName()] = self.metrics[i].getDictResult()
        return result

#########################################################################


class IncorrectOutput():
    def __init__(self, columns_name: str):
        '''
        Args:
            columns_name: 
        '''
        
        self.columns_name = columns_name
        columns = ['file_name']
        for i in range(len(columns_name)):
            columns.append(columns_name[i])
        columns.append("predict")
        self.df = pd.DataFrame(columns=columns)

    def update(self, output, yb, name_b):
        '''
        '''
        scores, pred_b = torch.max(output.data,1)
        index = torch.nonzero((pred_b != yb)).squeeze().tolist()
        if not isinstance(index, list):
            index = [index]  # index가 단일 값인 경우에 리스트로 변환하여 처리
        pred_b = pred_b.numpy()
        scores = scores.numpy()
        output = list(output.numpy())
        name_b = list(name_b)
        for i in index:
            data = {'file_name':name_b[i]}
            # class 개수 1개면 문제 생겨서 나눔.
            if len(output[0]) != 1:
                for j in range(len(output[0])):
                    data[self.columns_name[j]] = output[i][j]
            else:
                data[self.columns_name[0]] = output[i]
            data['score'] = scores[i]
            data['predict'] = pred_b[i]
            new_row = pd.DataFrame(data=data, index=['file_name'])
            self.df = pd.concat([self.df,new_row], ignore_index=True)

    def logMetric(self, epoch: int, filename: str = "incorrect_output.csv"):
        '''
        '''
        if not os.path.exists('temp'):
            os.mkdir('temp')
        self.df.to_csv(f'temp/{filename}.csv')
        mlflow.log_artifact(f'temp/{filename}.csv', f'output_epoch_{epoch}')

class ConfusionMatrix():
    '''
    Confusion Matrix for classification
    '''
    def __init__(self):
        '''
        '''
        self.conf_pred = []
        self.conf_label = []


    def update(self, output, yb):
        '''
        '''
        predictions_conv = output.numpy()
        labels_conv = yb.numpy()
        self.conf_pred.append(predictions_conv)
        self.conf_label.append(labels_conv)


    def logMetric(self, epoch: int):
        '''
        '''
        new_pred = np.concatenate(self.conf_pred)
        new_label = np.concatenate(self.conf_label)
        con_mat=sklearn.metrics.confusion_matrix(new_label, new_pred)

        cfs=sns.heatmap(con_mat,annot=True)
        cfs.set(title='Confusion Matrix', ylabel='True lable', xlabel='Predict label')
        figure = cfs.get_figure()
        
        # Save the plot as an image
        mlflow.log_figure(figure, f"output_epoch_{epoch}/confusion_matrix.jpg")
        
        #close figure
        plt.clf()
        
        

class OutputLog():
    '''
    Only available in regression model
    '''
    def __init__(self, columns_name: str, num_classes: int):
        '''
        '''
        self.num_classes = num_classes
        self.columns_name = columns_name
        columns = ['file_name']
        for i in range(num_classes):
            columns.append('predict ' + columns_name[i])
            columns.append('label ' + columns_name[i])
        self.df = pd.DataFrame(columns=columns)


    def update(self, output, yb, name_b):
        '''
        '''
        output = output.detach().cpu().numpy()
        yb = yb.cpu().numpy()

        output = list(output)
        yb = list(yb)
        name_b = list(name_b)
        for i in range(len(output)):
            data = {'file_name':name_b[i]}
            # class 개수 1개면 문제 생겨서 나눔.
            if self.num_classes != 1:
                for j in range(self.num_classes):
                    data['predict ' + self.columns_name[j]] = output[i][j]
                    data['label ' + self.columns_name[j]] = yb[i][j]
            else:
                data['predict ' + self.columns_name[0]] = output[i]
                data['label ' + self.columns_name[0]] = yb[i]
            new_row = pd.DataFrame(data=data, index=['file_name'])
            self.df = pd.concat([self.df,new_row], ignore_index=True)

    def logMetric(self, epoch: int, opt):
        '''
        '''
        if opt is None:
            if not os.path.exists('temp'):
                os.mkdir('temp')
            self.df.to_csv('temp/valid_output_data.csv')
            mlflow.log_artifact('temp/valid_output_data.csv', f'output_epoch_{epoch}')

        else:
            if not os.path.exists('temp'):
                os.mkdir('temp')
            self.df.to_csv('temp/train_output_data.csv')
            mlflow.log_artifact('temp/train_output_data.csv', f'output_epoch_{epoch}')
            
    def getOutputLog(self):
        '''
        '''
        return self.df

