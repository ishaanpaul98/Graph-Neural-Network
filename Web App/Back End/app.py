from NN.mpgnn import MPGNN

# Example usage
model = MPGNN(
    num_features=64,      # Number of input features per node
    hidden_channels=128,  # Size of hidden layers
    num_classes=1,        # Binary recommendation
    num_layers=2         # Number of message passing layers
)

# For training
output = model(x, edge_index)

# For making recommendations
predictions = model.predict(x, edge_index)