import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import numpy as np
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)
torch.backends.cudnn.deterministic = True  # 保证每次结果一样
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
import os
from collections import OrderedDict
import random
from scipy import sparse
import string
from stop_words import get_stop_words    # download stop words package from https://pypi.org/project/stop-words/
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
##########################################################
def proSeqToOnehot(proseq):
    dict = {
    'A': '100000000000000000000',
    'G': '010000000000000000000',
    'V': '001000000000000000000',
    'I': '000100000000000000000',
    'L': '000010000000000000000',
    'F': '000001000000000000000',
    'P': '000000100000000000000',
    'Y': '000000010000000000000',
    'M': '000000001000000000000',
    'T': '000000000100000000000',
    'S': '000000000010000000000',
    'H': '000000000001000000000',
    'N': '000000000000100000000',
    'Q': '000000000000010000000',
    'W': '000000000000001000000',
    'R': '000000000000000100000',
    'K': '000000000000000010000',
    'D': '000000000000000001000',
    'E': '000000000000000000100',
    'C': '000000000000000000010',
    'X': '000000000000000000001',
    'B': '000000000000000000000',
    'U': '000000000000000000000',
    'Z': '000000000000000000000',
    'O': '000000000000000000000'}
    proOnehot = []
    AlaOnehot = []
    for Ala in proseq:
        if dict[Ala]:
            for item in dict[Ala]:
                if item=='1':
                    AlaOnehot.append(1.0)
                else:
                    AlaOnehot.append(0.0)
            AlaOnehotcopy = AlaOnehot[:]
            AlaOnehot.clear()
            proOnehot.append(AlaOnehotcopy)
    return proOnehot

def preprocessing1(data):
    stop_words = get_stop_words('english')
    idx_data = []
    text_data = []
    label_data = []
    for idx,seq,text,label,score in data:
        s=' '.join(text)
        ss=s.translate(str.maketrans('','',string.punctuation))
        text_data.append(ss)
        label_data.append(label)
        idx_data.append(idx)
    label_data = np.array(label_data)
    idx_data = np.array(idx_data)

    u=defaultdict(int)
    for i in range(0,len(text_data)):
        line=text_data[i].strip('\n').split()
        for j in line:
            u[j]=u[j]+1

    u2=defaultdict(int)
    for i in u:
        if i.isdigit()==False:
            if u[i]>10:
                if i not in stop_words:
                    u2[i]=u[i]

    notesvocab={}
    for j in u2:
        if (u2[j]!=0) and (j.lower() not in notesvocab):
            notesvocab[j.lower()]=len(notesvocab)

    return idx_data,label_data,notesvocab

def similiar_score(score,training_idx,Label):
    Feature1 = score[:,training_idx]
    fcol=Feature1.sum(axis=1)
    for i in range(0, len(fcol)):
        if fcol[i]==0:
            fcol[i]=1
    num1=Feature1.shape[1]
    Feature_sum=repmat(fcol,num1,1)
    Feature_sum=Feature_sum.T
    guiyiFeature=Feature1/Feature_sum
    predicted=np.dot(guiyiFeature,Label)
    return predicted

def preprocessing(data,batchsize,training_idx,training_label,notesvocab):
    append = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    new_data = []
    lenlist = []
    for idx, sqe, text, label, score in data:
        s=' '.join(text)
        ss=s.translate(str.maketrans('','',string.punctuation))
        line=ss.strip('\n').split()
        temp=[]
        for j in line:
            if j in notesvocab:
                temp.append(j)
        # print(temp)
        sss=' '.join(temp)
        lenlist.append(len(sqe))
        new_data.append((sqe,temp,label,score,sss))
        # print(sss)
    new_data = np.array(new_data)
    sortlen = sorted(range(len(lenlist)), key=lambda k: lenlist[k])
    new_data = new_data[sortlen]

    batch_data = []
    for start_ix in range(0, len(new_data) - batchsize + 1, batchsize):
        thisblock = new_data[start_ix:start_ix + batchsize]
        mybsize = len(thisblock)
        # pro_text = []
        numsqe = 2000
        pro_seq = []
        for i in range(mybsize):
            # pro_text.append(thisblock[i][1])
            pro_onehot = proSeqToOnehot(thisblock[i][0])
            if len(thisblock[i][0]) >= numsqe:
                pro_onehotcopy = pro_onehot[0:2000]
            else:
                pro_onehotcopy = pro_onehot[:]
                for i in range(2000 - len(pro_onehot)):
                    appendcopy = append[:]
                    pro_onehotcopy.append(appendcopy)
            pro_seq.append(pro_onehotcopy)
         
        pro_text = []
        for i in range(mybsize):
            pro_text.append(thisblock[i][4])
        vect = TfidfVectorizer(min_df=1,vocabulary=notesvocab,binary=True)
        binaryn = vect.fit_transform(pro_text)
        binaryn=binaryn.A
        binaryn=np.array(binaryn,dtype=float)
        
        
        # print(thisblock[0][1])
        numword=np.max([len(ii[1]) for ii in thisblock])
        main_matrix = np.zeros((mybsize, numword), dtype= np.int)
        for i in range(main_matrix.shape[0]):
            for j in range(main_matrix.shape[1]):
                try:
                    if thisblock[i][1][j] in notesvocab:
                        main_matrix[i,j] = notesvocab[thisblock[i][1][j]]
                    
                except IndexError:
                    pass       # because initialze with 0, so you pad with 0
        
        yyy = []
        for ii in thisblock:
            yyy.append(ii[2])
        
        zzz = []
        for ii in thisblock:
            sc=ii[3].todense()
            zzz.append(sc)
        zzz = np.array(zzz)
        zzz = np.squeeze(zzz)
        zzz_new = similiar_score(zzz,training_idx,training_label)
        batch_data.append((autograd.Variable(torch.FloatTensor(pro_seq)),autograd.Variable(torch.from_numpy(main_matrix)), autograd.Variable(torch.FloatTensor(yyy)),autograd.Variable(torch.FloatTensor(zzz_new)),autograd.Variable(torch.FloatTensor(binaryn))))
    return batch_data

def l2norm(X, dim, eps=1e-8):
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X


class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=21, out_channels=hidden_dim, kernel_size=4)  # acid 21  21*2000 = 100*1998
        self.conv2 = nn.Conv1d(in_channels=21, out_channels=hidden_dim, kernel_size=8)  # acid 21  21*2000 = 100*1998
        self.conv3 = nn.Conv1d(in_channels=21, out_channels=hidden_dim, kernel_size=16)  # acid 21  21*2000 = 100*1998

        self.embed_drop = nn.Dropout(p=0.2)
        # self.word_embeddings = nn.Embedding(voclen+1, Embeddingsize, padding_idx=0)
        self.word_embeddings = nn.Embedding(voclen, Embeddingsize, padding_idx=0)
        self.convs1 = nn.Conv1d(Embeddingsize,hidden_dim,4)
        self.convs2 = nn.Conv1d(Embeddingsize,hidden_dim,8)
        self.convs3 = nn.Conv1d(Embeddingsize,hidden_dim,16)
        # self.H=nn.Linear(hidden_dim, 2 )
        self.W=nn.Parameter(torch.Tensor(batchsize,2*hidden_dim,3))
        nn.init.uniform_(self.W, -0.1, 0.1)
        self.fc = nn.Linear(2*hidden_dim, 3)
        self.layernorm = nn.LayerNorm(3)

        self.tanh = nn.Tanh()

        self.fc1 = nn.Linear(3*hidden_dim, classnum)
        self.dro = nn.Dropout(0.5)
        self.BN = nn.BatchNorm1d(3*hidden_dim,momentum=0.5)
        self.relu = nn.LeakyReLU(0.2)
        #Normalization
        self.hidden = self.init_hidden()
        self.lstm = nn.LSTM(Embeddingsize, hidden_dim)
        self.fc2 = nn.Linear(voclen, classnum)
        self.BN1 = nn.BatchNorm1d(voclen,momentum=0.5)
        self.fc11 = nn.Linear(voclen, classnum)
        self.fc22 = nn.Linear(classnum, classnum)
        self.W1=nn.Parameter(torch.Tensor(batchsize,classnum,3))
        self.fcn = nn.Linear(3, 1)

    def init_hidden(self):
        return (autograd.Variable(torch.zeros(1, batchsize, hidden_dim).cuda()),
                autograd.Variable(torch.zeros(1, batchsize, hidden_dim)).cuda())


    def forward(self,x,x1,score,x2):
        x = x.transpose(1,2)
        x = self.embed_drop(x)
        out1 = self.conv1(x)
        out1 = nn.MaxPool1d(out1.size()[2])(out1)
        out2 = self.conv2(x)
        out2 = nn.MaxPool1d(out2.size()[2])(out2)
        out3 = self.conv3(x)
        out3 = nn.MaxPool1d(out3.size()[2])(out3)
        out = torch.cat([out1,out2,out3], 2)


        thisembeddings=self.word_embeddings(x1)
        thisembeddings = self.embed_drop(thisembeddings)
        thisembeddings=thisembeddings.transpose(1, 2)
        out11=self.convs1(thisembeddings)
        out11=nn.MaxPool1d(out11.size()[2])(out11)
        out22=self.convs2(thisembeddings)
        out22=nn.MaxPool1d(out22.size()[2])(out22)
        out33=self.convs3(thisembeddings)
        out33=nn.MaxPool1d(out33.size()[2])(out33)
        outp = torch.cat([out11,out22,out33], 2)


        gate = torch.sigmoid(self.W*torch.cat([out,outp],dim=1))
        c_gate = torch.matmul(gate,out.transpose(1, 2))+torch.matmul((1-gate),outp.transpose(1, 2))
        output = self.layernorm(self.fc(self.dro(c_gate).transpose(1,2)))

        output = output.view(32, -1)
        outt = self.BN(output)
        outt = torch.sigmoid(self.fc1(outt))

        s1 = self.BN1(x2)
        score1 = self.fc2(s1)
        score1 = torch.sigmoid(score1)

        xx = self.fcn(torch.stack([outt,score1,score],dim=2))
        xx = xx.view(32, -1)
        xx = torch.sigmoid(xx)
        return xx


topk = 10

def trainmodel(model,batchtraining_data,batchval_data,loss_function,optimizer):
    print('start_training')
    modelsaved = []
    modelperform = []
    topk = 10

    bestresults = -1
    bestiter = -1
    for epoch in range(5000):
        model.train()

        lossestrain = []
        recall = []
        for mysentence in batchtraining_data:
            model.zero_grad()
            # model.hidden = model.init_hidden()
            targets = mysentence[2].cuda()
            tag_scores = model(mysentence[0].cuda(),mysentence[1].cuda(),mysentence[3].cuda(),mysentence[4].cuda())
            loss = loss_function(tag_scores, targets)
            loss.backward()
            optimizer.step()
            lossestrain.append(loss.data.mean())
        print(epoch)
        modelsaved.append(copy.deepcopy(model.state_dict()))
        print("XXXXXXXXXXXXXXXXXXXXXXXXXXXX")
        model.eval()

        recall = []
        for inputs in batchval_data:
            # model.hidden = model.init_hidden()
            targets = inputs[2].cuda()
            tag_scores = model(inputs[0].cuda(),inputs[1].cuda(),inputs[3].cuda(),inputs[4].cuda())

            loss = loss_function(tag_scores, targets)

            targets = targets.data.cpu().numpy()
            tag_scores = tag_scores.data.cpu().numpy()

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

        print('validation top-', topk, np.mean(recall))

        modelperform.append(np.mean(recall))
        if modelperform[-1] > bestresults:
            bestresults = modelperform[-1]
            bestiter = len(modelperform) - 1

        if (len(modelperform) - bestiter) > 5:
            print(modelperform, bestiter)
            return modelsaved[bestiter]

def testmodel(modelstate,batchtest_data,GOterm):
    model = ConvNet()
    model.cuda()
    model.load_state_dict(modelstate)
    loss_function = nn.BCELoss()
    model.eval()
    recall = []
    lossestest = []

    y_true = []
    y_scores = []

    for inputs in batchtest_data:
        # model.hidden = model.init_hidden()
        targets = inputs[2].cuda()

        tag_scores = model(inputs[0].cuda(),inputs[1].cuda(),inputs[3].cuda(),inputs[4].cuda())

        loss = loss_function(tag_scores, targets)

        targets = targets.data.cpu().numpy()
        tag_scores = tag_scores.data.cpu().numpy()

        lossestest.append(loss.data.mean())
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
    y_true = np.concatenate(y_true, axis=0)
    y_scores = np.concatenate(y_scores, axis=0)
    y_true = y_true.T
    y_scores = y_scores.T
    np.save('./results/y_true_DeepDF_'+GOterm, y_true)
    np.save('./results/y_scores_DeepDF_'+GOterm, y_scores)

def DeepDF(datatrain,datatest,GOterm,funGO):
    training_data, val_data = train_test_split(datatrain, test_size=0.1, random_state=42)
    del datatrain
    global batchsize
    batchsize=32
    global classnum
    classnum = len(funGO)
    global hidden_dim
    hidden_dim=200
    global Embeddingsize
    Embeddingsize=100
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") # cuda
    training_idx,training_label,notesvocab = preprocessing1(training_data)
    global voclen
    voclen = len(notesvocab)
    batchtraining_data = preprocessing(training_data,batchsize,training_idx,training_label,notesvocab)
    print('traindata ok ')
    batchtest_data = preprocessing(datatest,batchsize,training_idx,training_label,notesvocab)
    batchval_data = preprocessing(val_data,batchsize,training_idx,training_label,notesvocab)
    del training_data
    del val_data
    del training_idx
    del training_label
    del notesvocab
    model = ConvNet()
    model.cuda()
    loss_function = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(),lr=0.0003)
    basemodel = trainmodel(model,batchtraining_data,batchval_data,loss_function,optimizer)
    print('DeepDF_model alone: ')
    testmodel(basemodel,batchtest_data,GOterm)




# if __name__=="__main__":









