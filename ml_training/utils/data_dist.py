import matplotlib.pyplot as plt
import mlflow
import seaborn as sns

def logDatasetHistogram(train_df, val_df, columns, bins=10, width=0.25):
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