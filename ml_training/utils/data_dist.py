import matplotlib.pyplot as plt
import mlflow
import seaborn as sns

# +
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
    



# +
# def datasetKDE(train_df, val_df, grade, name):
#     fig = plt.figure(figsize=(30,20))
#     i=0
#     for g in grade:
#         for n in name:
#             i+=1
#             plt.subplot(4,5,i)
#             g_val_df = val_df[val_df['Rank'] == g]
#             g_train_df = train_df[train_df['Rank'] == g]
#             sns.kdeplot(g_val_df[n], bw='0.1')
#             sns.kdeplot(g_train_df[n], bw='0.1')
#             plt.xlabel('Value')
#             plt.ylabel('Density')
#             plt.title(f'Density Plot of {g} {n}')
#             plt.legend(labels=['valid', 'train'])
#     mlflow.log_figure(fig, "Dataset_KDE.jpg")
#     plt.clf()

# +
# def outputKDE(df, name, epoch):
#     df['grade'] = df['file_name'].str.split('_').str[3]
#     grade = ['1++', '1+', '2', '3']
#     fig = plt.figure(figsize=(30,5))
#     i=0
#     for n in name:
#       i += 1
#       plt.subplot(1,5,i)
#       sns.kdeplot(df[f'label {n}'],bw=0.1)
#       sns.kdeplot(df[f'predict {n}'],bw=0.1)
#       plt.xlabel('Value')
#       plt.ylabel('Density')
#       plt.title(f'Density Plot of {n}')
#       plt.legend(labels=['label', 'predict'])
#     mlflow.log_figure(fig, f'output_epoch_{epoch}/output(all)_KDE.jpg')
#     plt.clf()

#     fig = plt.figure(figsize=(30,20))
#     i=0
#     for g in grade:
#       for n in name:
#         i+=1
#         plt.subplot(4,5,i)
#         g_df = df[df['grade'] == g]
#         sns.kdeplot(g_df[f'label {n}'],bw=0.1)
#         sns.kdeplot(g_df[f'predict {n}'],bw=0.1)
#         plt.xlabel('Value')
#         plt.ylabel('Density')
#         plt.title(f'Density Plot of {g} {n}')
#         plt.legend(labels=['label', 'predict'])
#     mlflow.log_figure(fig, f'output_epoch_{epoch}/output(grade)_KDE.jpg')
#     plt.clf()