import pandas as pd
import numpy as np
import sklearn
import torch
import os
import mlflow
import seaborn as sns
import matplotlib.pyplot as plt


class Metrics():
    '''
    #1. 클래스명: Metrics \n
    #2. 목적/용도: 수치적인 metric을 선언하고 업데이트하는 클래스. accuracy, r2score, mae가 있음.\n
    '''
    def __init__(self,eval_function, num_classes, algorithm, data_length, columns_name):
        '''
        #1. 함수명: __init__ \n
        #2. 목적/용도: 실제 metric 클래스를 선언함\n 
        #3. Input parameters:\n
        eval_function: 사용할 metric의 이름\n
        num_classes: 예측 class 개수\n
        algorithm: regression/classification\n
        data_length: 총 데이터 개수\n
        columns_name: 예측 class의 열 이름\n
        '''
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
        '''
        #1. 함수명: update \n
        #2. 목적/용도: 선언한 metric 클래스들을 업데이트함\n 
        #3. Input parameters:\n
        output: 예측값\n
        yb: 정답값\n
        '''
        for metric in self.metrics:
            metric.update(output, yb)
        
    def getMetrics(self):
        '''
        #1. 함수명: getMetrics \n
        #2. 목적/용도: 현재 선언된 metric 클래스들을 반환함\n
        #3. Output: 리스트에 담긴 metric 클래스들. 
        '''
        return self.metrics
    
    def logMetrics(self, mode, epoch):
        '''
        #1. 함수명: logMetrics \n
        #2. 목적/용도: mlflow에 현재 metric 클래스들의 결과를 저장함.\n
        #3. Input parameters:\n
        mode: train/valid\n
        epoch: 현재 반복 횟수\n
        '''
        for metric in self.metrics:
            metric.logMetric(mode, epoch)
            
    def getResults(self):
        '''
        #1. 함수명: getResults \n
        #2. 목적/용도: 현재 metric 클래스들의 결과를 리스트에 담아 반환함.\n
        #3. Output: 리스트에 담긴 metric 클래스들의 결과\n
        '''
        result = []
        for metric in self.metrics:
            result.append(metric.getResult())
        return result 
    
    def getDictResults(self):
        '''
        #1. 함수명: getDictResults \n
        #2. 목적/용도: 현재 metric 클래스들의 결과를 dictionary에 { 클래스 이름: 결과 } 형태로 반환함.\n
        #3. Output: dictionary에 담긴 metric 클래스들의 결과\n
        '''
        result = {}
        for i in range(len(self.metrics)):
            result[self.metrics[i].getClassName()] = self.metrics[i].getDictResult()
        return result
    
class Accuracy():
    '''
    #1. 클래스명: Accuracy \n
    #2. 목적/용도: 정확도를 계산하는 클래스\n
    #3. 기타 참고사항: classification과 regression 둘다 사용 가능. 
    '''
    def __init__(self, length, num_classes, algorithm, columns_name, method = None):
        '''
        #1. 함수명: __init__ \n
        #2. 목적/용도: 실제 Accuracy 클래스를 선언함\n 
        #3. Input parameters:\n
        length: 총 데이터 개수\n
        num_classes: 예측 class 개수\n
        algorithm: regression/classification\n
        columns_name: 예측 class의 열 이름\n
        method: 정확도를 계산할 때 예측값과 정답값에 어떤 조작을 가할지 정하는 것. 현재는 round/floor/none이 있음.
        '''
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
        result = self.getResult()
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
    '''
    #1. 클래스명: R2score \n
    #2. 목적/용도: R2 score 계산하는 클래스\n 
    #3. 기타 참고사항: regression만 사용 가능. 
    '''
    def __init__(self, length):
        '''
        #1. 함수명: __init__ \n
        #2. 목적/용도: 실제 R2score 클래스를 선언함\n 
        #3. Input parameters:\n
        length: 총 데이터 개수\n
        '''
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
    '''
    #1. 클래스명: MeanAbsError \n
    #2. 목적/용도: MAE를 계산하는 클래스\n
    #3. 기타 참고사항: regression만 사용 가능. 
    '''
    def __init__(self, length, num_classes, columns_name):
        '''
        #1. 함수명: __init__ \n
        #2. 목적/용도: 실제 MeanAbsError 클래스를 선언함\n 
        #3. Input parameters:\n
        length: 총 데이터 개수\n
        num_classes: 예측 class 개수\n
        columns_name: 예측 class의 열 이름\n
        '''
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



#########################################################################


class IncorrectOutput():
    '''
    #1. 클래스명: IncorrectOutput \n
    #2. 목적/용도: 예측을 틀린 이미지와 그 예측값을 csv파일에 저장하는 클래스\n
    #3. 기타 참고사항: classification만 사용 가능. 
    '''
    def __init__(self, class_name: str):
        '''
        #1. 함수명: __init__ \n
        #2. 목적/용도: 실제 IncorrectOutput 클래스를 선언함\n 
        #3. Input parameters:\n
        length: 총 데이터 개수\n
        class_name: 예측 class의 unique 값들의 이름\n
        '''
        self.class_name = class_name
        columns = ['file_name']
        for i in range(len(class_name)):
            columns.append(class_name[i])
        columns.append("predict")
        self.df = pd.DataFrame(columns=columns)

    def update(self, output, yb, name_b):
        '''
        #1. 함수명: update \n
        #2. 목적/용도: 예측값과 정답값을 비교하여 틀린 값을 dataFrame에 저장함.\n 
        #3. Input parameters:\n
        output: 예측값\n
        yb: 정답값\n
        name_b: 이미지 파일 이름.
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
                    data[self.class_name[j]] = output[i][j]
            else:
                data[self.class_name[0]] = output[i]
            data['score'] = scores[i]
            data['predict'] = pred_b[i]
            new_row = pd.DataFrame(data=data, index=['file_name'])
            self.df = pd.concat([self.df,new_row], ignore_index=True)

    def logMetric(self, epoch: int, filename: str = "incorrect_output.csv"):
        '''
        #1. 함수명: logMetric \n
        #2. 목적/용도: mlflow에 현재까지의 dataframe을 csv형태로 저장함.\n 
        #3. Input parameters:\n
        epoch: 현재 반복 횟수\n
        filename: 저장할 csv 파일 이름\n
        '''
        if not os.path.exists('temp'):
            os.mkdir('temp')
        self.df.to_csv(f'temp/{filename}.csv')
        mlflow.log_artifact(f'temp/{filename}.csv', f'output_epoch_{epoch}')

class ConfusionMatrix():
    '''
    #1. 클래스명: ConfusionMatrix \n
    #2. 목적/용도: confunsion matrix를 그리는 클래스\n
    #3. 기타 참고사항: classification만 사용 가능. 
    '''
    def __init__(self):
        '''
        #1. 함수명: __init__ \n
        #2. 목적/용도: 실제 ConfusionMatrix 클래스를 선언함\n 
        '''
        self.conf_pred = []
        self.conf_label = []


    def update(self, output, yb):
        '''
        #1. 함수명: update \n
        #2. 목적/용도: 예측값과 정답값을 비교하여 confusion matrix 제작.\n 
        #3. Input parameters:\n
        output: 예측값\n
        yb: 정답값\n
        '''
        predictions_conv = output.numpy()
        labels_conv = yb.numpy()
        self.conf_pred.append(predictions_conv)
        self.conf_label.append(labels_conv)


    def logMetric(self, epoch: int):
        '''
        #1. 함수명: logMetric \n
        #2. 목적/용도: mlflow에 현재까지의 confusion matrix를 저장함.\n 
        #3. Input parameters:\n
        epoch: 현재 반복 횟수\n
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
    #1. 클래스명: OutputLog \n
    #2. 목적/용도: 모델의 학습 과정에서 모델의 예측값을 모두 기록하는 클래스.\n 
    '''
    def __init__(self, columns_name: str, num_classes: int):
        '''
        #1. 함수명: __init__ \n
        #2. 목적/용도: dataframe의 column을 정의함.\n 
        #3. Input parameters:\n
        columns_name: 예측값들의 column name\n
        num_classes: 예측 class 의 개수
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
        #1. 함수명: update \n
        #2. 목적/용도: 모델의 예측값과 결과값을 기존의 dataframe에 이어붙임.\n 
        #3. Input parameters:\n
        output: 모델의 예측값\n
        yb: 정답값\n
        name_b: 이미지의 파일명\n
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
        #1. 함수명: logMetric \n
        #2. 목적/용도: 만든 dataframe을 mlflow에 저장함.\n 
        #3. Input parameters:\n
        epoch: 현재 반복 횟수\n
        opt: train과 valid를 구분하는 값.\n
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
        #1. 함수명: getOutputLog \n
        #2. 목적/용도: 만든 dataframe을 반환함.\n 
        '''
        return self.df

