import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, GATConv, GCNConv
from torch_geometric.utils import add_self_loops, degree
import math

class AttentionGNNConv(MessagePassing):
    """
    Enhanced message passing layer with attention mechanism
    """
    def __init__(self, in_channels, out_channels, heads=4, dropout=0.1):
        super(AttentionGNNConv, self).__init__(aggr='add')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.dropout = dropout
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=in_channels,
            num_heads=heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Feature transformation
        self.lin = nn.Linear(in_channels, out_channels)
        self.norm = nn.LayerNorm(out_channels)
        
    def forward(self, x, edge_index):
        # Add self-loops
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        
        # Compute attention weights
        row, col = edge_index
        edge_weights = self.compute_attention_weights(x, row, col)
        
        # Start propagating messages with attention
        return self.propagate(edge_index, x=x, edge_weights=edge_weights)
    
    def compute_attention_weights(self, x, row, col):
        # Simple attention based on feature similarity
        src_features = x[row]
        dst_features = x[col]
        
        # Compute cosine similarity
        similarity = F.cosine_similarity(src_features, dst_features, dim=1)
        attention_weights = F.softmax(similarity, dim=0)
        
        return attention_weights
    
    def message(self, x_j, edge_weights):
        # Apply attention weights to messages
        return edge_weights.view(-1, 1) * self.lin(x_j)
    
    def update(self, aggr_out, x):
        # Residual connection and normalization
        out = self.norm(aggr_out + self.lin(x))
        return F.relu(out)

class EnhancedMPGNN(nn.Module):
    """
    Enhanced Message Passing Graph Neural Network with advanced features
    """
    def __init__(self, num_user_features, num_movie_features, hidden_channels, 
                 num_classes=1, num_layers=3, dropout=0.2, heads=4, 
                 use_attention=True, use_residual=True):
        super(EnhancedMPGNN, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_attention = use_attention
        self.use_residual = use_residual
        
        # Enhanced feature transformations
        self.user_encoder = nn.Sequential(
            nn.Linear(num_user_features, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels),
            nn.LayerNorm(hidden_channels)
        )
        
        self.movie_encoder = nn.Sequential(
            nn.Linear(num_movie_features, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels),
            nn.LayerNorm(hidden_channels)
        )
        
        # Multiple types of graph convolution layers
        if use_attention:
            self.conv_layers = nn.ModuleList([
                AttentionGNNConv(hidden_channels, hidden_channels, heads, dropout)
                for _ in range(num_layers)
            ])
        else:
            self.conv_layers = nn.ModuleList([
                GCNConv(hidden_channels, hidden_channels)
                for _ in range(num_layers)
            ])
        
        # Skip connections
        self.skip_connections = nn.ModuleList([
            nn.Linear(hidden_channels, hidden_channels)
            for _ in range(num_layers - 1)
        ])
        
        # Enhanced edge prediction with multiple layers
        self.edge_predictor = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, num_classes)
        )
        
        # Movie popularity bias
        self.popularity_bias = nn.Linear(1, 1)
        
    def forward(self, x_user, x_movie, edge_index, movie_popularity=None):
        # Encode user and movie features
        x_user = self.user_encoder(x_user)
        x_movie = self.movie_encoder(x_movie)
        
        # Combine features
        x = torch.cat([x_user, x_movie], dim=0)
        x_initial = x.clone()
        
        # Message passing layers with skip connections
        for i, conv in enumerate(self.conv_layers):
            x_new = conv(x, edge_index)
            
            if self.use_residual and i > 0:
                # Skip connection
                x_new = x_new + self.skip_connections[i-1](x)
            
            x = x_new
        
        # Get user and movie embeddings
        user_emb = x[:x_user.size(0)]
        movie_emb = x[x_user.size(0):]
        
        # Get edge embeddings
        edge_emb = torch.cat([
            user_emb[edge_index[0]],
            movie_emb[edge_index[1]]
        ], dim=1)
        
        # Predict edge ratings
        out = self.edge_predictor(edge_emb)
        
        # Add popularity bias if available
        if movie_popularity is not None:
            popularity_bias = self.popularity_bias(movie_popularity.unsqueeze(1))
            out = out + popularity_bias
        
        return out
    
    def predict(self, x_user, x_movie, edge_index, movie_popularity=None):
        """
        Make predictions for recommendation
        """
        self.eval()
        with torch.no_grad():
            out = self.forward(x_user, x_movie, edge_index, movie_popularity)
            return torch.sigmoid(out)

class MultiTaskMPGNN(nn.Module):
    """
    Multi-task GNN that predicts both rating and watch probability
    """
    def __init__(self, num_user_features, num_movie_features, hidden_channels, 
                 num_layers=3, dropout=0.2):
        super(MultiTaskMPGNN, self).__init__()
        
        # Shared encoder
        self.shared_encoder = EnhancedMPGNN(
            num_user_features, num_movie_features, hidden_channels,
            num_classes=hidden_channels, num_layers=num_layers, dropout=dropout
        )
        
        # Task-specific heads
        self.rating_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, 1)
        )
        
        self.watch_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, 1)
        )
        
    def forward(self, x_user, x_movie, edge_index, movie_popularity=None):
        # Get shared embeddings
        shared_emb = self.shared_encoder.forward(x_user, x_movie, edge_index, movie_popularity)
        
        # Task-specific predictions
        rating_pred = self.rating_head(shared_emb)
        watch_pred = self.watch_head(shared_emb)
        
        return rating_pred, watch_pred
    
    def predict(self, x_user, x_movie, edge_index, movie_popularity=None):
        """
        Make predictions for recommendation
        """
        self.eval()
        with torch.no_grad():
            rating_pred, watch_pred = self.forward(x_user, x_movie, edge_index, movie_popularity)
            # Combine predictions (you can adjust the weights)
            combined_score = 0.7 * torch.sigmoid(rating_pred) + 0.3 * torch.sigmoid(watch_pred)
            return combined_score 