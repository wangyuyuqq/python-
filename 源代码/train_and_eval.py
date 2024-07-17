import time
import numpy as np
from torch import nn
import time
import os
import torch
import logging
from torch.optim import AdamW
from transformers import Trainer, TrainingArguments, BertTokenizer, BertModel, BertPreTrainedModel, BertConfig, \
    get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader
from transformers.utils.notebook import format_time
from modeling import BertForSeq
from data_process import InputDataSet,read_data,fill_paddings
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#可以用分布式
def train(batch_size,EPOCHS):#每一组有多少个句子进行训练和训练轮数
    best_acc = 0
    model = BertForSeq.from_pretrained('bert-base-chinese')
    train = read_data('data/train.csv')
    val = read_data('data/dev.csv')
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    train_dataset = InputDataSet(train, tokenizer, 128)
    val_dataset = InputDataSet(val, tokenizer, 128)

    train_dataloader = DataLoader(train_dataset,batch_size)
    val_dataloader = DataLoader(val_dataset,batch_size)

    optimizer = AdamW(model.parameters(), lr=2e-5)  #bert官方优化器
    total_steps = len(train_dataloader) * EPOCHS  # 总步数等于长度乘以轮数
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=total_steps)#调度器
    total_t0 = time.time()
#控制台打印一下数据，打印到log文件中
    log = log_creater(output_dir='./cache/logs/')
#打印信息
    log.info("   Train batch size = {}".format(batch_size))
    log.info("   Total steps = {}".format(total_steps))
    log.info("   Training Start!")
#对训练轮次进行循环
    for epoch in range(EPOCHS):
        total_train_loss = 0
        t0 = time.time()
        model.to(device)
        model.train()
        for step, batch in enumerate(train_dataloader):
#先将特征投到设备中
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['labels'].to(device)
            model.zero_grad()#先进行梯度清零
#求模型的输出
            outputs = model(input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids,labels=labels)
            loss = outputs.loss
            total_train_loss += loss.item()# item才是数值
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)#避免梯度爆炸，进行梯度剪裁，防止参数大于一导致不断传递，底层是贝叶斯
            optimizer.step()
            scheduler.step()
        avg_train_loss = total_train_loss / len(train_dataloader)#求平均损失
        train_time = format_time(time.time() - t0)
        log.info('====Epoch:[{}/{}] avg_train_loss={:.5f}===='.format(epoch+1,EPOCHS,avg_train_loss))
        log.info('====Training epoch took: {:}===='.format(train_time))
        log.info('Running Validation...')
        model.eval()
        #验证集的损失和准确率
        avg_val_loss, avg_val_acc = evaluate(model, val_dataloader)
        val_time = format_time(time.time() - t0)
        log.info('====Epoch:[{}/{}] avg_val_loss={:.5f} avg_val_acc={:.5f}===='.format(epoch+1,EPOCHS,avg_val_loss,avg_val_acc))
        log.info('====Validation epoch took: {:}===='.format(val_time))
        log.info('')
        if avg_val_acc > best_acc:
            best_acc=avg_val_acc
            torch.save(model,'./cache/model_stu.bin2')
            print('Model Saved!')
    log.info('')
    log.info('Training Completed!')
    print('Total training took{:} (h:mm:ss)'.format(format_time(time.time() - total_t0)))
def evaluate(model,val_dataloader):
    total_val_loss = 0
    corrects = []
    for batch in val_dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        labels = batch['labels'].to(device)
        with torch.no_grad():
            outputs = model(input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids,labels=labels)
        logits = torch.argmax(outputs.logits,dim=1) #先进行softmax
        ## 把每个batch预测的准确率加入到一个list中
        ## 在加入之前，preds和labels变成cpu的格式,防止报错
        preds = logits.detach().cpu().numpy()
        labels_ids = labels.to('cpu').numpy()
        corrects.append((preds == labels_ids).mean())  #求出准确率   得到每一个batch准确率的列表
        ## 返回loss
        loss = outputs.loss
        ## 把每个batch的loss加入 total_val_loss
        ## 总共有len(val_dataloader)个batch
        total_val_loss += loss.item()
    #求平均得到平均准确率和loss
    avg_val_loss = total_val_loss / len(val_dataloader)
    avg_val_acc = np.mean(corrects)
    return avg_val_loss, avg_val_acc
#这个函数打印信息
def log_creater(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    log_name = '{}.log'.format(time.strftime('%Y-%m-%d-%H-%M'))
    final_log_file = os.path.join(output_dir, log_name)
    # creat a log
    log = logging.getLogger('train_log')
    log.setLevel(logging.DEBUG)
    # FileHandler
    file = logging.FileHandler(final_log_file)
    file.setLevel(logging.DEBUG)
    # StreamHandler
    stream = logging.StreamHandler()
    stream.setLevel(logging.DEBUG)
    # Formatter
    formatter = logging.Formatter(
        '[%(asctime)s][line: %(lineno)d] ==> %(message)s')
    # setFormatter
    file.setFormatter(formatter)
    stream.setFormatter(formatter)
    # addHandler
    log.addHandler(file)
    log.addHandler(stream)
    log.info('creating {}'.format(final_log_file))
    return log
if __name__ == '__main__':
    train(batch_size=20,EPOCHS=10)
