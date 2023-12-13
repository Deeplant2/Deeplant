import matplotlib.pyplot as plt
import mlflow
import seaborn as sns

def logDatasetHistogram(train_df, val_df, columns, bins=10, width=0.25):
    '''
    #1. 함수명: logDatasetHistogram \n
    #2. 목적/용도: 학습 시작 전, 현재 학습에 사용될 dataset의 히스토그램을 그린 후 mlflow에 저장한다.\n 
    #3. Input parameters:\n
    train_df = 학습에 사용될 데이터 프레임.\n
    val_df = 검증에 사용될 데이터 프레임.\n
    columns = 히스토그램을 그릴 데이터가 적힌 열\n
    bins = 히스토그램을 그릴 때 막대의 개수\n
    width = 막대의 width\n
    #4. Output : 모델 객체를 반환한다.\n
    #5. 기타 참고사항\n
    '''
    fig = plt.figure(figsize=(10,5))
    for column in columns:
        plt.subplot(1,2,1)
        plt.hist(train_df[column], bins=bins, width=width)
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.title(f'Histogram of Train {column}')
        
        plt.subplot(1,2,2)
        plt.hist(val_df[column], bins=bins, width=width)
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.title(f'Histogram of Valid {column}')
        mlflow.log_figure(fig, f"dataset/Histogram_{column}.jpg")
        plt.clf()
        
def logDatasetKDE(train_df, val_df, columns, bw=0.1):
    '''
    #1. 함수명: logDatasetHistogram \n
    #2. 목적/용도: 학습 시작 전, 현재 학습에 사용될 dataset의 KDE를 그린 후 mlflow에 저장한다.\n 
    #3. Input parameters:\n
    train_df = 학습에 사용될 데이터 프레임.\n
    val_df = 검증에 사용될 데이터 프레임.\n
    columns = 히스토그램을 그릴 데이터가 적힌 열\n
    bw = KDE를 그릴 때 얼마나 세밀하게 그릴 지에 대한 값\n
    #4. Output : 모델 객체를 반환한다.\n
    #5. 기타 참고사항\n
    '''
    fig = plt.figure(figsize=(10,5))
    for column in columns:
        plt.subplot(1,2,1)
        sns.kdeplot(train_df[column], bw=bw)
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.title(f'Histogram of Train {column}')
        
        plt.subplot(1,2,2)
        sns.kdeplot(val_df[column], bw=bw)
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.title(f'Histogram of Valid {column}')
        mlflow.log_figure(fig, f"dataset/KDE_{column}.jpg")
        plt.clf()