import pandas as pd

data = pd.read_csv(r'D:\Python大作业\train.csv')
# 随机打乱数据，按照8：2划分训练集和测试集
data = data.sample(frac=1)
train_data = data[:int(0.80 * len(data))]
dev_data = data[int(0.80 * len(data)):]
# 保存训练集和验证集为两个不同的csv文件
train_data.to_csv('data/train.csv', index=False)
dev_data.to_csv('data/dev.csv', index=False)
'''
data2 = pd.read_csv('data/dev.csv')
data2 = data2.sample(frac=1)
data = pd.read_csv('data/train.csv')
data = data.sample(frac=1)
data.to_csv('data/train.csv', index=False)
data2.to_csv('data/dev.csv', index=False)
'''
