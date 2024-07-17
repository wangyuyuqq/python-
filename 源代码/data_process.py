import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from transformers import Trainer,TrainingArguments, BertTokenizer, BertModel, BertPreTrainedModel,BertConfig
from torch.utils.data import Dataset, DataLoader
from torch import nn
import warnings
warnings.filterwarnings('ignore')
#递归限制，防止内存溢出
import sys
sys.setrecursionlimit(3000)
def read_data(data_dir):
    #导入文本清除空值
    data = pd.read_csv(data_dir)
    data['Ofiicial Account Name'] = data['Ofiicial Account Name'].fillna('')
    data['text'] = data['Title']
    '''with open(r'D:\2212046wy\data\stopwords.txt', "r", encoding="utf-8") as f:
        # 读取文件中的每一行，去除空白字符，并且过滤掉空字符串
        lines = f.readlines()
        lines = [line.strip() for line in lines]
        stopwords = [word for word in lines if word != ""]
    data=read_data(train_dir)
    for i in range(len(data)):
        for word in stopwords:
            data['text'][i] = data['text'][i].replace(word, "")
    print(data)'''
    return data
def fill_paddings(data, maxlen):
    #补全句子长度的函数
    if len(data) < maxlen:
        pad_len = maxlen-len(data)
        paddings = [0 for _ in range(pad_len)]
        data = torch.tensor(data + paddings)
    else:
        data = torch.tensor(data[:maxlen])
    return data
class InputDataSet():
    def __init__(self,data,tokenizer,max_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
    def __len__(self,):
        return len(self.data)
    def __getitem__(self, item):  # item是索引 用来取数据
        text = str(self.data['text'][item])
        labels = self.data['label'][item]
        labels = torch.tensor(labels, dtype=torch.long)
    #构建分词
        #这里先用tokenize将文本分词，然后用convert_tokens_to_ids函数将每个词对应编码，生成词汇表
        tokens = self.tokenizer.tokenize(text)
        tokens_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        tokens_ids = [101] + tokens_ids + [102]
        #将分好词后的序列补充到最大长度
        input_ids = fill_paddings(tokens_ids,self.max_len)
        #构建好一个注意力掩码序列
        attention_mask = [1 for _ in range(len(tokens_ids))]
        attention_mask = fill_paddings(attention_mask,self.max_len)
        #用来表示输入序列中的不同句子的边界
        token_type_ids = [0 for _ in range(len(tokens_ids))]
        token_type_ids = fill_paddings(token_type_ids,self.max_len)
        return {
            'text':text,
            'input_ids':input_ids,
            'attention_mask':attention_mask,
            'token_type_ids':token_type_ids,
            'labels':labels
        }
#用来定义测试集的输入格式，因为测试集没有label
class InputDataSet2():
    def __init__(self,data,tokenizer,max_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
    def __len__(self,):
        return len(self.data)
    def __getitem__(self, item):  # item是索引 用来取数据
        text = str(self.data['text'][item])
    #构建分词
        #这里先用tokenize将文本分词，然后用convert_tokens_to_ids函数将每个词对应编码，生成词汇表
        tokens = self.tokenizer.tokenize(text)
        tokens_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        tokens_ids = [101] + tokens_ids + [102]
        #将分好词后的序列补充到最大长度
        input_ids = fill_paddings(tokens_ids,self.max_len)
        #构建好一个注意力掩码序列
        attention_mask = [1 for _ in range(len(tokens_ids))]
        attention_mask = fill_paddings(attention_mask,self.max_len)
        #用来表示输入序列中的不同句子的边界
        token_type_ids = [0 for _ in range(len(tokens_ids))]
        token_type_ids = fill_paddings(token_type_ids,self.max_len)
        return {
            'text':text,
            'input_ids':input_ids,
            'attention_mask':attention_mask,
            'token_type_ids':token_type_ids,

        }
if __name__ == '__main__':
    train_dir = 'data/test.csv'
    dev_dir = 'data/dev.csv'
    model_dir = 'bert-base-chinese'
    train = read_data(train_dir)
    test = read_data(dev_dir)
    tokenizer = BertTokenizer.from_pretrained(model_dir)
    train_dataset = InputDataSet2(train,tokenizer=tokenizer, max_len=16)
    #DataLoader函数引用自torch.utils.data模块
    #将train_dataset这个数据集对象封装为一个可迭代的数据加载器对象，存储在train_dataloader变量中。设置批次大小为4，即每次从数据加载器中取出4个数据元素
    train_dataloader = DataLoader(train_dataset,batch_size=4)
    batch = next(iter(train_dataloader))
    #从train_dataloader这个可迭代对象中取出第一个元素
    print(batch)
    print(batch['input_ids'].shape)
    print(batch['attention_mask'].shape)
    print(batch['token_type_ids'].shape)








