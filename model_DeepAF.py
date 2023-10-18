import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import numpy as np
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)
torch.backends.cudnn.deterministic = True  
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
import copy
import math
from torch.nn import Parameter
import scipy.io as scio
import torch.nn.functional as F
import codecs
from numpy.matlib import repmat
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from transformers import AdamW, BertConfig
from pytorch_pretrained_bert import BertTokenizer,BertModel
# from pytorch_pretrained_bert.optimization import BertAdam
# from pytorch_pretrained_bert import BertTokenizer,BertModel
# from pytorch_pretrained_bert.optimization import BertAdam
# from transformers import AdamW
import os
from collections import OrderedDict
import random
from scipy import sparse
import string
from stop_words import get_stop_words    # download stop words package from https://pypi.org/project/stop-words/
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import esm
import tqdm
# from Evaluation import *
from torch.nn import Conv1d
import torch.utils.data as Data
from torch.autograd import Variable
##########################################################
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") # cuda
attention_dropout = 0.1
maxlen=512
class MyDataSet(Data.Dataset):
    def __init__(self, data,prot_fea,GOterm,species):
        # print('data',data.shape)
        new_data, lenlist= [], []
        for id,name,sqe,text,label in data:
            temp=[]
            temp1=[]
            count=1
            for a in text:
                temp.append(a.strip('\n'))
                file4 = './data/'+species[0:-1]+'/text/'+GOterm+'_'+str(id)+'_'+str(count)
                input_text=torch.load(file4)
                temp1.append(input_text)
                count=count+1
            lenlist.append(len(text))
            output = torch.cat(temp1,dim=0)
            new_data.append((sqe,temp,output,label))
        input_seq,input_textbiobert,target_labels= [], [], []
        for sqe, text, output, label in new_data:
            input_textbiobert.append(output)
            target_labels.append(label)
        self.input_seq = torch.FloatTensor(prot_fea)
        self.input_textbiobert = input_textbiobert
        self.target_labels = target_labels
    
    def __len__(self):
        return len(self.input_seq)
 
    def __getitem__(self, idx):
        return self.input_seq[idx], self.input_textbiobert[idx],self.target_labels[idx]

def collate_fn(examples):
    device = torch.device('cuda:0') 
    textnum=np.max([ex[1].size(0) for ex in examples])
    input_seq,input_textbiobert,target_labels= [], [], []
    for ex in examples:
        input_seq.append(ex[0].unsqueeze(0))
        target_labels.append(ex[2])
        if ex[1].size(0) < textnum:
            temp=torch.zeros((textnum-ex[1].size(0)),768).to(device)
            a=torch.cat([ex[1],temp],dim=0)
        else:
            a=ex[1]
        input_textbiobert.append(a.unsqueeze(0))

    output = torch.cat(input_seq,dim=0)
    output1 = torch.cat(input_textbiobert,dim=0)
    input_seq = torch.FloatTensor(output)
    input_textbiobert = output1
    target_labels = torch.FloatTensor(target_labels)
    return input_seq,input_textbiobert,target_labels



class Attention(nn.Module):
    def __init__(self, in_size, hidden_size=16):
        super(Attention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z)
        beta = torch.softmax(w, dim=1)
        return (beta * z).sum(1), beta

class ConvNet(nn.Module):
    def __init__(self,alpha=0.25,gamma=2, reduction='mean', lb_smooth=0.1, ignore_index=-100):
        super().__init__()
        self.BN = nn.BatchNorm1d(768, momentum=0.5)
        # self.bert=BertModel.from_pretrained('/home/DUTIR/zhaoyingwen/BioBert/biobert_v1.1_pubmed_v2.5.1_convert')
        # self.bert=self.bert.to(DEVICE)
        self.hidden_dim = 768
        self.hidden_dim1 = 768
        self.classnum = classnum
        self.sqe = nn.Linear(768, 768)
        self.text = nn.Linear(768, 768)
        self.sqe_fc = nn.Linear(self.hidden_dim, self.classnum)
        self.text_fc = nn.Linear(self.hidden_dim, self.classnum)
        self.T_block1 = I_S_Block(self.sqe_fc, self.text_fc, self.hidden_dim)
        self.T_block2 = I_S_Block(self.sqe_fc, self.text_fc, self.hidden_dim)
        self.T_block3 = I_S_Block(self.sqe_fc, self.text_fc, self.hidden_dim)
        self.fc = nn.Linear(2*self.hidden_dim1, self.classnum)
        self.criterion = nn.BCELoss()  # nn.BCELoss()  nn.BCEWithLogitsLoss()
        self.criterion1 = nn.BCELoss()  # nn.BCELoss()  nn.BCEWithLogitsLoss()

        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.crit = nn.BCEWithLogitsLoss(reduction='none')
        self.lb_smooth = lb_smooth
        self.lb_ignore = ignore_index
        self.log_softmax = nn.LogSoftmax(dim=1)

        self.dropout = nn.Dropout(0.5)
        self.attention = Attention(768)
        self.fc1 = nn.Linear(2*768, self.classnum)#+voclen,768

    def forward_logit(self,x_s,x_b):
        x_s=self.sqe(x_s)
        x_b=self.text(x_b)
        H_S, H_T = self.T_block1(x_s.unsqueeze(1), x_b)
        out_emb, att = self.attention((H_T+x_b))
        logits_all = torch.hstack([(H_S+ x_s.unsqueeze(1)).squeeze(1),out_emb])
        out = self.fc1(logits_all)
        predicted = torch.sigmoid(out)
        return predicted

    def loss1(self, logits, label):
        loss = self.criterion(logits, label)
        return loss


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class SelfOutput(nn.Module):
    def __init__(self, hidden_size, hidden_dropout_prob):
        super(SelfOutput, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states



class I_S_Block(nn.Module):
    def __init__(self, intent_emb, slot_emb, hidden_size):
        super(I_S_Block, self).__init__()
        self.I_S_Attention = I_S_SelfAttention(hidden_size, 2 * hidden_size, hidden_size)
        self.I_Out = SelfOutput(hidden_size, attention_dropout)
        self.S_Out = SelfOutput(hidden_size, attention_dropout)

    def forward(self, H_intent_input, H_slot_input):
        H_slot, H_intent = self.I_S_Attention(H_intent_input, H_slot_input)
        H_slot = self.S_Out(H_slot, H_slot_input)
        H_intent = self.I_Out(H_intent, H_intent_input)

        return H_intent, H_slot

class I_S_SelfAttention(nn.Module):
    def __init__(self, input_size, hidden_size, out_size):
        super(I_S_SelfAttention, self).__init__()

        self.num_attention_heads = 8
        self.attention_head_size = int(hidden_size / self.num_attention_heads)

        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.out_size = out_size
        self.query = nn.Linear(input_size, self.all_head_size)
        self.query_slot = nn.Linear(input_size, self.all_head_size)
        self.key = nn.Linear(input_size, self.all_head_size)
        self.key_slot = nn.Linear(input_size, self.all_head_size)
        self.value = nn.Linear(input_size, self.out_size)
        self.value_slot = nn.Linear(input_size, self.out_size)
        self.dropout = nn.Dropout(attention_dropout)

    def transpose_for_scores(self, x):
        last_dim = int(x.size()[-1] / self.num_attention_heads)
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, last_dim)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, intent, slot):
        # extended_attention_mask = mask.unsqueeze(1).unsqueeze(2)

        # extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        # attention_mask = (1.0 - extended_attention_mask) * -10000.0
        mixed_query_layer = self.query(intent)
        mixed_key_layer = self.key(slot)
        mixed_value_layer = self.value(slot)
        mixed_query_layer_slot = self.query_slot(slot)
        mixed_key_layer_slot = self.key_slot(intent)
        mixed_value_layer_slot = self.value_slot(intent)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        query_layer_slot = self.transpose_for_scores(mixed_query_layer_slot)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        key_layer_slot = self.transpose_for_scores(mixed_key_layer_slot)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        value_layer_slot = self.transpose_for_scores(mixed_value_layer_slot)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_scores_slot = torch.matmul(query_layer_slot, key_layer_slot.transpose(-1, -2))
        attention_scores_slot = attention_scores_slot / math.sqrt(self.attention_head_size)
        attention_scores_intent = attention_scores #+ attention_mask

        attention_scores_slot = attention_scores_slot #+ attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs_slot = nn.Softmax(dim=-1)(attention_scores_slot)
        attention_probs_intent = nn.Softmax(dim=-1)(attention_scores_intent)

        attention_probs_slot = self.dropout(attention_probs_slot)
        attention_probs_intent = self.dropout(attention_probs_intent)

        context_layer_slot = torch.matmul(attention_probs_slot, value_layer_slot)
        context_layer_intent = torch.matmul(attention_probs_intent, value_layer)

        context_layer = context_layer_slot.permute(0, 2, 1, 3).contiguous()
        context_layer_intent = context_layer_intent.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.out_size,)
        new_context_layer_shape_intent = context_layer_intent.size()[:-2] + (self.out_size,)

        context_layer = context_layer.view(*new_context_layer_shape)
        context_layer_intent = context_layer_intent.view(*new_context_layer_shape_intent)
        return context_layer, context_layer_intent


topk = 10

def trainmodel(model,train_loader,loss_function,optimizer,accumulation_steps = 8):
    print('start_training')
    for i in range(10):
        model.train()
        model.zero_grad()
        for batch_idx, (input_seq, input_textb,target_labels) in enumerate(train_loader):
            input_seq,input_textb, target_labels = Variable(torch.FloatTensor(input_seq)).to(DEVICE), Variable(input_textb).to(DEVICE),Variable(torch.FloatTensor(target_labels)).to(DEVICE)
            logits = model.forward_logit(input_seq,input_textb)
            loss = model.loss1(logits,target_labels)

        # 梯度积累
            loss = loss/accumulation_steps
            loss.backward()

            if((batch_idx+1) % accumulation_steps) == 0:
            # 每 4 次更新一下网络中的参数
                optimizer.step()
                optimizer.zero_grad()
                model.zero_grad()

            if ((batch_idx+1) % accumulation_steps) == 1:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss:{:.6f}'.format(
                    i+1, batch_idx, len(train_loader), 100. *
                    batch_idx/len(train_loader), loss.item()
                ))
    return copy.deepcopy(model.state_dict())


def testmodel(modelstate,test_loader,GOterm,fold,AA,train_index):
    model = ConvNet()
    model.cuda()
    model.load_state_dict(modelstate)
    # loss_function = nn.BCELoss()
    model.eval()
    recall = []

    y_true = []
    y_scores = []

    for batch_idx, (input_seq,input_textb, target_labels) in enumerate(test_loader):

        input_seq, input_textb,target_labels = torch.FloatTensor(input_seq).to(DEVICE), Variable(input_textb).to(DEVICE),torch.FloatTensor(target_labels).to(DEVICE)
        tag_scores= model.forward_logit(input_seq, input_textb)

        targets = target_labels.data.cpu().numpy()
        tag_scores = tag_scores.data.cpu().numpy()

        y_true.append(targets)
        y_scores.append(tag_scores)

        for iii in range(0, len(tag_scores)):
            temp = {}
            for iiii in range(0, len(tag_scores[iii])):
                temp[iiii] = tag_scores[iii][iiii]
            temp1 = [(k, temp[k]) for k in sorted(temp, key=temp.get, reverse=True)]
            thistop = int(np.sum(targets[iii]))
            hit = 0.0

            for ii in temp1[0:max(thistop, topk)]:
                if targets[iii][ii[0]] == 1.0:
                    hit = hit + 1
            if thistop != 0:
                recall.append(hit / thistop)
    print('test top-', topk, np.mean(recall))
    y_true = np.concatenate(y_true, axis=0)
    y_scores = np.concatenate(y_scores, axis=0)
    y_true = y_true.T
    y_scores = y_scores.T
    scio.savemat(file_name="./results/DeepAF"+"_fold:"+str(fold)+"_"+AA+GOterm+'.mat', mdict={'real':y_true,'predicted':y_scores,'idx':train_index})

def DeepAF(data,prot_fea,GOterm,AA,batch):
    global classnum
    classnum = data[0][4].shape[0]
    global batchsize
    batchsize=batch
    global num
    num=len(data)
    kf_5 = KFold(n_splits=5, shuffle=True, random_state=0)
    fold = 0
    for train_index, test_index in kf_5.split(data):
        train_loader = Data.DataLoader(MyDataSet(data[train_index],prot_fea[train_index],GOterm,AA), batch_size=batchsize, collate_fn=collate_fn, shuffle=True,drop_last=True)
        test_loader = Data.DataLoader(MyDataSet(data[test_index],prot_fea[test_index],GOterm,AA), batch_size=batchsize, collate_fn=collate_fn, shuffle=True,drop_last=True)
        print('processing data ok ')
        model = ConvNet()
        model.cuda()
        loss_function = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(),lr=0.0003)
        basemodel = trainmodel(model,train_loader,loss_function,optimizer)
        print('DeepAF alone: ',fold)
        testmodel(basemodel,test_loader,GOterm,fold,AA,train_index)
        fold = fold + 1
        torch.cuda.empty_cache()
        #exit()



# if __name__=="__main__":









