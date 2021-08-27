import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import numpy as np
torch.manual_seed(1)
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
import copy
import math
from torch.nn import Parameter
import scipy.io as scio
from scipy import sparse
import torch.nn.functional as F
import codecs
from numpy.matlib import repmat
from sklearn.model_selection import train_test_split
import os
from collections import OrderedDict
import random
import pickle
from model_DeepDF import DeepDF
##########################################################
f = open("./2016data/2016GOtermsnew_bmc.pkl", "rb")
terms = pickle.load(f)
f = open('./2016data/2016train.pkl','rb')
datatrain = pickle.load(f)
f = open('./2016data/2016test.pkl','rb')
datatest = pickle.load(f)
file='./2016data/2016Alltrain_Scores.mat'
Score_train = scio.loadmat(file)['scores']
file='./2016data/2016Alltest_Scores.mat'
Score_test = scio.loadmat(file)['scores']

for i in range(0, Score_train.shape[0]):
    Score_train[i,i]=0

GOnames=['bp','mf','cc']
for i in range(2,-1,-1):
    a=GOnames[i]
    print(a)
    termtemp=terms.iloc[0].iat[i]
    GOAtrain=np.zeros(shape=(datatrain.shape[0],len(termtemp)))
    for ii in range(0, datatrain.shape[0]):
        goa=datatrain.iloc[ii].iat[3]
        for iii in range(0,len(goa)):
            if goa[iii] in termtemp:
                index=termtemp.index(goa[iii])
                GOAtrain[ii][index]=1
    GOAtest=np.zeros(shape=(datatest.shape[0],len(termtemp)))
    for ii in range(0, datatest.shape[0]):
        goa=datatest.iloc[ii].iat[3]
        for iii in range(0,len(goa)):
            if goa[iii] in termtemp:
                index=termtemp.index(goa[iii])
                GOAtest[ii][index]=1
    mcol=GOAtrain.sum(axis=1)
    id_train=[k for k in range(GOAtrain.shape[0]) if mcol[k]>0]
    mcol=GOAtest.sum(axis=1)
    id_trest=[k for k in range(GOAtest.shape[0]) if mcol[k]>0]

    Score_trainnew = sparse.lil_matrix(sparse.csc_matrix(Score_train)[:,id_train])
    Score_testnew = sparse.lil_matrix(sparse.csc_matrix(Score_test)[:,id_train])

    datatrain1 = []
    protein_idx={}
    for j in range(0, datatrain.shape[0]):
        if j in id_train:
            protein_idx[datatrain.iloc[j].iat[0]]=len(protein_idx)
            datatrain1.append((protein_idx[datatrain.iloc[j].iat[0]],datatrain.iloc[j].iat[1],datatrain.iloc[j].iat[2],GOAtrain[j],Score_trainnew[j]))
    datatrain1 = np.array(datatrain1)
    print(datatrain1.shape)

    datatest1 = []
    protein_idx={}
    for j in range(0, datatest.shape[0]):
        if j in id_trest:
            protein_idx[datatest.iloc[j].iat[0]]=len(protein_idx)
            datatest1.append((protein_idx[datatest.iloc[j].iat[0]],datatest.iloc[j].iat[1],datatest.iloc[j].iat[2],GOAtest[j],Score_testnew[j]))
    datatest1 = np.array(datatest1)
    print(datatest1.shape)

    del GOAtrain
    del GOAtest
    del Score_trainnew
    del Score_testnew

    DeepDF(datatrain1,datatest1,a,termtemp)





