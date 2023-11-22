import pandas as pd
import numpy as np
import sklearn
import torch
import os
import mlflow
import seaborn as sns
import matplotlib.pyplot as plt


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



# +
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
        predictions_conv = pred_b.numpy()
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
