import torch
import torch.nn as nn
import numpy as np
import dgl
from sklearn.cluster import KMeans
import faiss
import networkx as nx
import torch.nn.functional as F
from data_preprocessing import DataProcessor
from config import device
import copy
from collections import defaultdict
from numpy import linalg as LA


def hyG_function(g,G,com_DICT,coms_G,num_nodes):
    DICT=copy.deepcopy(com_DICT)
    LEN=len(DICT)
    n_hedge = LEN

    member_citing = []
    
    for community in DICT.keys():
        members = DICT[community]
        for member in members :
            member_citing.append([member, community])

    member_community = torch.LongTensor(member_citing)
    data_dict = {
            ('node', 'in', 'edge'): (member_community[:,0], member_community[:,1]),
            ('edge', 'con', 'node'): (member_community[:,1], member_community[:,0])
        }

    lst=[]
    for i in member_citing:
      lst.append(i[0])
    s=set(lst)
    s=len(s)
    num_nodes_dict = {'edge': LEN,'node':s}
    hyG = dgl.heterograph(data_dict,num_nodes_dict=num_nodes_dict)
    rows=g.number_of_nodes()
    columns=n_hedge

    len_rows=rows
    nl=np.eye(len_rows)
    nl=torch.from_numpy(nl)
    v_feat=nl

    len_edges=n_hedge
    nl=np.eye(len_edges)
    nl=torch.from_numpy(nl)
    e_feat=nl

    hyG.ndata['h'] = {'edge' : e_feat.type('torch.FloatTensor'), 'node' : v_feat.type('torch.FloatTensor')}
    e_feat = e_feat.type('torch.FloatTensor')
    v_feat=v_feat.type('torch.FloatTensor')


    # Find the number of keys each value appears in
    value_counts = defaultdict(int)
    for key, values in DICT.items():
        unique_values = set(values)
        for value in unique_values:
            value_counts[value] += 1
    
    # Calculate the score for each value
    total_keys = len(DICT)
    value_scores = {}
    uniqueness_list=[]
    for value, count in value_counts.items():
        score = 1 - (count / total_keys)
        value_scores[value] = score
        uniqueness_list.append(score)
    
    uniqueness = np.array(uniqueness_list)
    uniqueness = torch.LongTensor(uniqueness).to(device)

    centrality_values=list(nx.closeness_centrality(G).values())
    centrality_values = torch.LongTensor(centrality_values).to(device)
    
    building_edges=[]
    member_nodes=[]
    for i in DICT.keys():
      for j in DICT[i]:
        j=str(j)+'a'
        building_edges.append((j,i))
        member_nodes.append(j)
    member_nodes=set(member_nodes)
    
    
    B = nx.Graph()
    B.add_edges_from(building_edges)
    
    node=member_nodes
    from networkx.algorithms import bipartite
    G_B = bipartite.weighted_projected_graph(B, node)
    
    
    A=nx.to_numpy_array(G_B)
    degree=[]
    for i in range (len(A)):
      c=0
      for j in range (len(A)):
        c=c+A[i][j]
      degree.append(c)
    degree=np.array(degree)
    D=np.diag(degree)
    L=D-A
    
    '''
    import scipy.sparse as sp
    A = sp.csr_matrix(nx.adjacency_matrix(G_B))
    degree = A.sum(axis=1).A1
    D = sp.diags(degree)
    L = D - A
    '''
    '''
    L = nx.laplacian_matrix(G_B)
    L = csr_matrix(L)  # Convert to sparse matrix for efficient computation
    L = L.toarray()
    '''
    # Compute eigenvectors and eigenvalues
    eign_v, eign_vec = LA.eigh(L)
    
    # Perform random sign flipping
    for i in range(eign_vec.shape[1]):
        maximum_absolute_value_index = np.argmax(np.abs(eign_vec[:, i]))
        sign = np.sign(eign_vec[maximum_absolute_value_index, i])
        random_signs = np.random.choice([-1, 1], size=eign_vec.shape[0])
        eign_vec[:, i] *= sign * random_signs
    
    # Convert eigenvectors and eigenvalues to torch tensors
    eign_vec = torch.tensor(eign_vec)
    eign_vec = eign_vec.type(torch.FloatTensor).to(device)
    
    eign_v = torch.tensor(eign_v)
    eign_v = eign_v.type(torch.FloatTensor).to(device)
    
    v_feat=v_feat.to(device)
    e_feat=e_feat.to(device)
    hyG=hyG.to(device)
    return v_feat ,e_feat, hyG, LEN,coms_G,DICT,uniqueness,centrality_values,eign_vec,eign_v