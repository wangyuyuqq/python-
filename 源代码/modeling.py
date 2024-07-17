from data_process import read_data,InputDataSet
from transformers import Trainer,TrainingArguments, BertTokenizer, BertModel, BertPreTrainedModel,BertConfig
from torch.utils.data import Dataset, DataLoader
from torch import nn
from transformers.modeling_outputs import SequenceClassifierOutput
import torch
## 做句子的分类 BertForSequence
class BertForSeq(BertPreTrainedModel):
    def __init__(self,config):  ##  config.json
        super(BertForSeq,self).__init__(config)
        self.config = BertConfig(config)
        self.num_labels = 2           #二分类任务，因此label的种类为2
        self.bert = BertModel(config) #加载这个配置文件，获得bert模型
        self.dropout = nn.Dropout(config.hidden_dropout_prob)  #一个概率随机屏蔽神经元，防止神经网络过拟合
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)  #分类器，进行维度变换和线性变换，把输入的东西转化为标签类别
        self.init_weights()
    def forward(
            self,
            input_ids,
            attention_mask = None,
            token_type_ids = None,  #输入的embadding
            labels = None,
            return_dict = None
    ):#要加label
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict #false输出tensor 不设置输出string
        # loss损失和预测值preds
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=return_dict
        )  #调用bert模型把embedding传进去，出预测值
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        # logits是softmax层的输入
        #通过softmax解码，标签为（0，1）的概率,越靠近哪个输出哪个
        logits = self.classifier(pooled_output)
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()    #交叉熵损失,用到了softmax
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))  # 二分类的参数要view
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,  #损失
            logits=logits,  #softmax层的输入，可以理解为是个概率
            hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
        )

if __name__ == '__main__':
    ## 加载编码器和模型
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    model = BertForSeq.from_pretrained('bert-base-chinese')
    ## 准备数据
    dev = read_data('data/train.csv')
    dev_dataset = InputDataSet(dev,tokenizer=tokenizer,max_len=128)
    dev_dataloader = DataLoader(dev_dataset,batch_size=16,shuffle=False)
    ## 把数据做成batch
    batch = next(iter(dev_dataloader))
    ## 设置device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #有哪个用哪个
    ## 输入embedding
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    token_type_ids = batch['token_type_ids'].to(device)
    labels = batch['labels'].to(device)
    model.to(device)
    ## 预测
    model.eval()
    ## 得到输出
    outputs = model(input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids,labels=labels)
    ## 取输出里面的loss和logits
    logits = outputs.logits
    loss = outputs.loss
    print(logits)
    print("loss=",loss.item()) #计算loss的值
    preds = torch.argmax(logits,dim=1)
    print(preds)