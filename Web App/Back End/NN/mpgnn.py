import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

class GNNConv(MessagePassing):
    """
    Custom message passing layer for the GNN
    """
    def __init__(self, in_channels, out_channels):
        super(GNNConv, self).__init__(aggr='add')
        self.lin = nn.Linear(in_channels, out_channels)
        
    def forward(self, x, edge_index):
        # Add self-loops to the adjacency matrix
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        
        # Compute normalization
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        
        # Normalize node features
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        
        # Start propagating messages
        return self.propagate(edge_index, x=x, norm=norm)
    
    def message(self, x_j, norm):
        # Message computation
        return norm.view(-1, 1) * self.lin(x_j)

class MPGNN(nn.Module):
    """
    Message Passing Graph Neural Network for recommendation systems
    """
    def __init__(self, num_user_features, num_movie_features, hidden_channels, num_classes, num_layers=2, dropout=0.2):
        super(MPGNN, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # User and movie feature transformations
        self.user_lin = nn.Linear(num_user_features, hidden_channels)
        self.movie_lin = nn.Linear(num_movie_features, hidden_channels)
        
        # Message passing layers
        self.conv_layers = nn.ModuleList([
            GNNConv(hidden_channels, hidden_channels)
            for _ in range(num_layers)
        ])
        
        # Edge prediction layer
        self.edge_lin = nn.Linear(hidden_channels * 2, num_classes)
        
    def forward(self, x_user, x_movie, edge_index):
        # Transform user and movie features
        x_user = self.user_lin(x_user)
        x_movie = self.movie_lin(x_movie)
        
        # Combine features
        x = torch.cat([x_user, x_movie], dim=0)
        
        # Message passing layers
        for conv in self.conv_layers:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Get user and movie embeddings
        user_emb = x[:x_user.size(0)]
        movie_emb = x[x_user.size(0):]
        
        # Get edge embeddings by concatenating user and movie embeddings
        edge_emb = torch.cat([
            user_emb[edge_index[0]],
            movie_emb[edge_index[1]]
        ], dim=1)
        
        # Predict edge ratings
        out = self.edge_lin(edge_emb)
        
        return out
    
    def predict(self, x_user, x_movie, edge_index):
        """
        Make predictions for recommendation
        """
        self.eval()
        with torch.no_grad():
            out = self.forward(x_user, x_movie, edge_index)
            return torch.sigmoid(out)  # For binary classification/recommendation 
