import dgl
import copy
import bisect
from scipy.sparse import csr_matrix
import json
import networkx as nx
import networkx.algorithms.community as nx_comm
import torch
import numpy as np
from scipy.sparse import coo_matrix
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from networkx.readwrite import json_graph
from numpy import linalg as LA
import random
import scipy
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, recall_score,precision_score, accuracy_score,average_precision_score,precision_recall_curve,auc
from torch.nn import Parameter
import torch.optim as optim
import pdb
import itertools
from cdlib import algorithms
from torch_geometric.datasets import AttributedGraphDataset,LastFMAsia,Twitch,CitationFull
from collections import defaultdict
from torch_geometric.datasets import CitationFull

from data_preprocessing import DataProcessor  # Adjust the import path as necessary
from model_and_layers import THTN_attn,hyperedge_clustering_coefficient,GCN # Adjust the import path as necessary
from config import device
from hypergraph_construction import hyG_function

device = torch.device("cuda:1" if torch.cuda.is_available() else "CPU")

dataset_name = 'wiki'#[cora,citeseer,dblp,PubMed, wiki, LastFMAsia]
path_to_data = '/home/ksaifuddin1/Experiments/data'
path_to_label = f'/home/ksaifuddin1/Experiments/data/label_{dataset_name}.txt'
random_state=1

# Load and prepare data
data_processor = DataProcessor(path_to_data, path_to_label, dataset_name,random_state)
g, G, com_DICT,coms_G, edge_index, default_feat, node_features, number_class= data_processor.get_data_components()
label=data_processor.label
train_label = data_processor.train_label
val_label = data_processor.val_label
test_label = data_processor.test_label

#HyG construction
num_hyperedges = len(com_DICT) 
num_nodes = G.number_of_nodes()
v_feat ,e_feat, hyG, LEN,coms_G,DICT,uniqueness,centrality_values,eign_vec,eign_v = hyG_function(g,G,com_DICT,coms_G,num_nodes)

node_feat=default_feat
gcn_model=GCN(node_feat.shape[1],512,LEN)
g=g.add_self_loop()

size_of_coms_G=len(coms_G)
dict_hyperedge_clustering_coefficient={}
for i in range (size_of_coms_G):
  dict_hyperedge_clustering_coefficient[i]=hyperedge_clustering_coefficient(coms_G[i])




"""Train"""
model =THTN_attn(v_feat.shape[1], 64, LEN, LEN,  number_class, 0.5,4,coms_G,LEN,centrality_values,uniqueness,eign_vec,e_feat,DICT,dict_hyperedge_clustering_coefficient,G)
model=model.to(device)
loss_fn = nn.CrossEntropyLoss()

#optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
optimizer = torch.optim.Adam(itertools.chain(model.parameters(), gcn_model.parameters()), lr=0.01)
best_val_acc = 0
patience=0

file_path = f'/home/ksaifuddin1/Hypergraph Transformer/thtn/outputs/{dataset_name}.txt'
with open(file_path, 'a') as file:
    for i in range(200):
        model.train()
        pred= model(hyG, v_feat, e_feat,centrality_values,uniqueness,eign_vec,node_feat,g,gcn_model, True,True)
        train_pred, test_pred= train_test_split( pred,test_size=0.25, random_state=random_state)
        size=int(len(label)*0.25)
        val_pred=train_pred[0:size]
        train_pred=train_pred[size:]
    
        loss = loss_fn(train_pred, train_label)
        pred_cls = torch.argmax(train_pred, -1)
        train_acc = torch.eq(pred_cls, train_label).sum().item()/len(train_label)
    
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
        with torch.no_grad():
    
          model.eval()
          val_cls = torch.argmax(val_pred, -1)
          val_acc = torch.eq(val_cls, val_label).sum().item()/len(val_label)
    
          # Save the best validation accuracy and the corresponding test accuracy.
          if best_val_acc < val_acc:
            best_val_acc = val_acc
            E=i
            patience=0
            file_path = f'/home/ksaifuddin1/Hypergraph Transformer/thtn/pth_folder/latest_{dataset_name}.pth'
            torch.save(test_pred, file_path)
          else:
            patience+=1
          if patience==100:
            break
        if i % 10 == 0:
          train_statement='In epoch {}, train loss: {:.4f}, val_acc: {:.4f} (best_val_acc: {:.4f})'.format(i, loss, val_acc, best_val_acc)
          print(train_statement)
          file.write(train_statement+'\n')

    file_path = f'/home/ksaifuddin1/Hypergraph Transformer/thtn/pth_folder/latest_{dataset_name}.pth'
    best_test_pred = torch.load(file_path)
    with torch.no_grad():
      test_pred=best_test_pred
      test_cls = torch.argmax(test_pred, -1)
      test_acc = torch.eq(test_cls, test_label).sum().item()/len(test_label)
      test_statement='Result: Best Epoch: {}, Accuracy: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, F1-score {:.4f}'.format(E,accuracy_score(test_label.cpu(),test_cls.cpu()),precision_score(test_label.cpu(),test_cls.cpu(),average='weighted'),recall_score(test_label.cpu(),test_cls.cpu(),average='weighted'),f1_score(test_label.cpu(),test_cls.cpu(),average='weighted'))
      print(test_statement)
      file.write(test_statement+'\n')