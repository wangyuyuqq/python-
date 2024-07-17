import torch
import numpy
import pandas as pd
from torch.optim import AdamW
from modeling import BertForSeq
from data_process import InputDataSet,read_data,fill_paddings
from data_process import read_data,InputDataSet
from transformers import Trainer,TrainingArguments, BertTokenizer, BertModel, BertPreTrainedModel,BertConfig
from torch.utils.data import Dataset, DataLoader
from torch import nn
from transformers.modeling_outputs import SequenceClassifierOutput
import torch
from modeling import BertForSeq
from data_process import InputDataSet,InputDataSet2,read_data,fill_paddings
model_name = 'bert-base-chinese'
model_path = './cache/model_stu.bin3'
# 加载分词器和模型
tokenizer = BertTokenizer.from_pretrained(model_name)
model = torch.load(model_path)
# 加载test.csv文件
test = read_data('data/test.csv')
test_dataset=InputDataSet2(test,tokenizer=tokenizer,max_len=64)
test_dataloader = DataLoader(test_dataset,batch_size=16,shuffle=False)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #有哪个用哪个
model.to(device)
model.eval()
corrects=[]
for batch in test_dataloader:
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    token_type_ids = batch['token_type_ids'].to(device)
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
    logits = torch.argmax(outputs.logits, dim=1)  # 先进行softmax
    ## 把每个batch预测的准确率加入到一个list中

    '''probabilities = torch.sigmoid(outputs.logits)
    threshold = 0.6
    # 最后根据概率值和阈值判断每个样本的预测结果
    logits = torch.argmax((probabilities >= threshold).long(),dim=1)
    ## 在加入之前，preds和labels变成cpu的格式,防止报错返回CPU'''
    preds = logits.detach().cpu().numpy()
    for i in preds:
        corrects.append(i)
#print(corrects)
'''# 导入numpy和sklearn.metrics模块
import numpy as np
from sklearn import metrics
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
#示例代码，导出准确率等评判便准
test_dataset=pd.read_csv(Answerpath)
print(accuracy_score(test_dataset['label'], corrects))
fpr, tpr, thresholds = metrics.roc_curve(test_dataset['label'], corrects, pos_label=1)
auc_value = metrics.auc(fpr, tpr)
print("AUC value", auc_value)
print(precision_recall_fscore_support(test_dataset['label'], corrects,average='binary'))'''
pds=pd.read_csv(r'C:\Users\ASUS\Desktop\submit_example.csv')
df=pd.DataFrame({'id':pds['id'],'label':corrects})
df.to_csv(r'C:\Users\ASUS\Desktop\submit_example.csv',index=False,sep=',')
#提示输出完毕
print("DONE!")