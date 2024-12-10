import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim, Tensor

from torch_geometric.utils import structured_negative_sampling
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn import LGConv

from sentence_transformers import SentenceTransformer
from torch_geometric.data import HeteroData
from torch_geometric.transforms import RandomLinkSplit, ToUndirected
from sklearn.model_selection import train_test_split
import pandas as pd
import torch
import sentence_transformers


class LightGCN(nn.Module):
    def __init__(self, num_users, num_items, num_layers=4, dim_h=64):
        super().__init__()
        
        self.num_users = num_users
        self.num_items = num_items
        self.num_layers = num_layers
        #Embeddings for users
        self.emb_users = nn.Embedding(num_embeddings=self.num_users, embedding_dim=dim_h)
        #Embeddings for content items
        self.emb_items = nn.Embedding(num_embeddings=self.num_items, embedding_dim=dim_h)
        #Convolutions layers
        self.convs = nn.ModuleList(LGConv() for _ in range(num_layers))

        #Initializing two normalized matrices for the weights for users and items
        nn.init.normal_(self.emb_users.weight, std=0.01)
        nn.init.normal_(self.emb_items.weight, std=0.01)

    def forward(self, edge_index):
        #Initializing the embeddings
        emb = torch.cat([self.emb_users.weight, self.emb_items.weight])
        #Converting it to a tensor
        embs = [emb]
        #looping through the convolutions and appending it to the embs tensor
        for conv in self.convs:
            emb = conv(x=emb, edge_index=edge_index)
            embs.append(emb)
        #Calculating the final embeddings
        emb_final = 1/(self.num_layers+1) * torch.mean(torch.stack(embs, dim=1), dim=1)
        #Splitting emb_final
        emb_users_final, emb_items_final = torch.split(emb_final, [self.num_users, self.num_items])
        
        return emb_users_final, self.emb_users.weight, emb_items_final, self.emb_items.weight