import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim

from torch_geometric.utils import structured_negative_sampling
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn import LGConv
from sklearn.model_selection import train_test_split
import pandas as pd
import torch

K = 20
LAMBDA = 1e-6
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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
    

def bpr_loss(emb_users_final, emb_users, emb_pos_items_final, emb_pos_items, emb_neg_items_final, emb_neg_items):
    reg_loss = LAMBDA * (emb_users.norm().pow(2) +
                        emb_pos_items.norm().pow(2) +
                        emb_neg_items.norm().pow(2))

    pos_ratings = torch.mul(emb_users_final, emb_pos_items_final).sum(dim=-1)
    neg_ratings = torch.mul(emb_users_final, emb_neg_items_final).sum(dim=-1)

    bpr_loss = torch.mean(torch.nn.functional.softplus(pos_ratings - neg_ratings))
    # bpr_loss = torch.mean(torch.nn.functional.logsigmoid(pos_ratings - neg_ratings))

    return -bpr_loss + reg_loss

def sample_mini_batch(edge_index, BATCH_SIZE = 1024):
    # Generate BATCH_SIZE random indices
    index = np.random.choice(range(edge_index.shape[1]), size=BATCH_SIZE)

    # Generate negative sample indices
    edge_index = structured_negative_sampling(edge_index)
    edge_index = torch.stack(edge_index, dim=0)
    
    user_index = edge_index[0, index]
    pos_item_index = edge_index[1, index]
    neg_item_index = edge_index[2, index]
    
    return user_index, pos_item_index, neg_item_index

def get_user_items(edge_index):
    user_items = dict()
    for i in range(edge_index.shape[1]):
        user = edge_index[0][i].item()
        item = edge_index[1][i].item()
        if user not in user_items:
            user_items[user] = []
        user_items[user].append(item)
    return user_items

def compute_recall_at_k(items_ground_truth, items_predicted):
    num_correct_pred = np.sum(items_predicted, axis=1)
    num_total_pred = np.array([len(items_ground_truth[i]) for i in range(len(items_ground_truth))])

    recall = np.mean(num_correct_pred / num_total_pred)

    return recall

def compute_ndcg_at_k(items_ground_truth, items_predicted):
    test_matrix = np.zeros((len(items_predicted), K))

    for i, items in enumerate(items_ground_truth):
        length = min(len(items), K)
        test_matrix[i, :length] = 1
    
    max_r = test_matrix
    idcg = np.sum(max_r * 1. / np.log2(np.arange(2, K + 2)), axis=1)
    dcg = items_predicted * (1. / np.log2(np.arange(2, K + 2)))
    dcg = np.sum(dcg, axis=1)
    idcg[idcg == 0.] = 1.
    ndcg = dcg / idcg
    ndcg[np.isnan(ndcg)] = 0.
    
    return np.mean(ndcg)

def get_metrics(model, edge_index, exclude_edge_indices):

    ratings = torch.matmul(model.emb_users.weight, model.emb_items.weight.T)

    for exclude_edge_index in exclude_edge_indices:
        user_pos_items = get_user_items(exclude_edge_index)
        exclude_users = []
        exclude_items = []
        for user, items in user_pos_items.items():
            exclude_users.extend([user] * len(items))
            exclude_items.extend(items)
        ratings[exclude_users, exclude_items] = -1024

    # get the top k recommended items for each user
    _, top_K_items = torch.topk(ratings, k=K)

    # get all unique users in evaluated split
    users = edge_index[0].unique()

    test_user_pos_items = get_user_items(edge_index)

    # convert test user pos items dictionary into a list
    test_user_pos_items_list = [test_user_pos_items[user.item()] for user in users]

    # determine the correctness of topk predictions
    items_predicted = []
    for user in users:
        ground_truth_items = test_user_pos_items[user.item()]
        label = list(map(lambda x: x in ground_truth_items, top_K_items[user]))
        items_predicted.append(label)

    recall = compute_recall_at_k(test_user_pos_items_list, items_predicted)
    ndcg = compute_ndcg_at_k(test_user_pos_items_list, items_predicted)

    return recall, ndcg

def test(model, edge_index, exclude_edge_indices):
    emb_users_final, emb_users, emb_items_final, emb_items = model.forward(edge_index)
    user_indices, pos_item_indices, neg_item_indices = structured_negative_sampling(edge_index, contains_neg_self_loops=False)

    emb_users_final, emb_users = emb_users_final[user_indices], emb_users[user_indices]

    emb_pos_items_final, emb_pos_items = emb_items_final[pos_item_indices], emb_items[pos_item_indices]
    emb_neg_items_final, emb_neg_items = emb_items_final[neg_item_indices], emb_items[neg_item_indices]

    loss = bpr_loss(emb_users_final, emb_users, emb_pos_items_final, emb_pos_items, emb_neg_items_final, emb_neg_items).item()

    recall, ndcg = get_metrics(model, edge_index, exclude_edge_indices)

    return loss, recall, ndcg

def training(BATCH_SIZE = 1024, NUM_EPOCHS = 20, PATH='model.pt'):
    _, ratings, num_users, num_items, _, _, edge_index = getEdgeIndices()
    train_index, train_edge_index, _, val_edge_index = getTrainTestValIndices(ratings, edge_index)
    model = LightGCN(num_users, num_items)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    n_batch = int(len(train_index)/BATCH_SIZE)

    for epoch in range(NUM_EPOCHS):
        model.train()

        for _ in range(n_batch):
            optimizer.zero_grad()

            emb_users_final, emb_users, emb_items_final, emb_items = model.forward(train_edge_index)

            user_indices, pos_item_indices, neg_item_indices = sample_mini_batch(train_edge_index)

            emb_users_final, emb_users = emb_users_final[user_indices], emb_users[user_indices]
            emb_pos_items_final, emb_pos_items = emb_items_final[pos_item_indices], emb_items[pos_item_indices]
            emb_neg_items_final, emb_neg_items = emb_items_final[neg_item_indices], emb_items[neg_item_indices]

            train_loss = bpr_loss(emb_users_final, emb_users, emb_pos_items_final, emb_pos_items, emb_neg_items_final, emb_neg_items)

            train_loss.backward()
            optimizer.step()
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss,
            }, PATH)
        if epoch % 5 == 0:
            model.eval()
            val_loss, recall, ndcg = test(model, val_edge_index, [train_edge_index])
            print(f"Epoch {epoch} | Train loss: {train_loss.item():.5f} | Val loss: {val_loss:.5f} | Val recall@{K}: {recall:.5f} | Val ndcg@{K}: {ndcg:.5f}")
    #torch.save(model, 'entire_model.pt', _use_new_zipfile_serialization=False)

def getEdgeIndices():
    movie_path = '../../Dataset/ml-latest-small/movies.csv'
    rating_path = '../../Dataset/ml-latest-small/ratings.csv'
    movies = pd.read_csv(movie_path)
    ratings = pd.read_csv(rating_path)
    ratings = ratings.loc[ratings['movieId'].isin(movies['movieId'].unique())]
    user_mapping = {userid: i for i, userid in enumerate(ratings['userId'].unique())}
    item_mapping = {isbn: i for i, isbn in enumerate(ratings['movieId'].unique())}
    # Count users and items
    num_users = len(user_mapping)
    num_items = len(item_mapping)
    num_total = num_users + num_items
    # Build the adjacency matrix based on user ratings
    user_ids = torch.LongTensor([user_mapping[i] for i in ratings['userId']])
    item_ids = torch.LongTensor([item_mapping[i] for i in ratings['movieId']])
    edge_index = torch.stack((user_ids, item_ids))
    
    return movies, ratings, num_users, num_items, user_ids, item_ids, edge_index

def getTrainTestValIndices(ratings, edge_index):
    # Create training, validation, and test adjacency matrices
    train_index, test_index = train_test_split(range(len(ratings)), test_size=0.2, random_state=0)
    val_index, test_index = train_test_split(test_index, test_size=0.5, random_state=0)

    train_edge_index = edge_index[:, train_index]
    val_edge_index = edge_index[:, val_index]
    test_edge_index = edge_index[:, test_index]

    return train_index, train_edge_index, test_edge_index, val_edge_index

def recommendationForNewUser(model, optimizer, newUser, likedMovieIds, likedRatings):
    if len(likedRatings) != len(likedMovieIds):
        print("Each movie needs to be rated!")
        return
    #Increase the number of users by 1
    #model.num_users += 1
    #Get recommendations 
    movies, ratings, _, _, _, _, _ = getEdgeIndices()
    ratings = ratings.loc[ratings['movieId'].isin(movies['movieId'].unique())]
    item_mapping = {isbn: i for i, isbn in enumerate(ratings['movieId'].unique())}
    userList = [newUser for i in likedMovieIds]
    userTensor = torch.LongTensor(userList)
    itemTensor = torch.LongTensor([item_mapping[i] for i in likedMovieIds])
    edge_index = torch.stack((userTensor, itemTensor))
    model.train()
    optimizer.zero_grad()
    emb_users_final, emb_users, emb_items_final, emb_items = model.forward(edge_index)
    user_indices, pos_item_indices, neg_item_indices = sample_mini_batch(edge_index)
    emb_users_final, emb_users = emb_users_final[user_indices], emb_users[user_indices]
    emb_pos_items_final, emb_pos_items = emb_items_final[pos_item_indices], emb_items[pos_item_indices]
    emb_neg_items_final, emb_neg_items = emb_items_final[neg_item_indices], emb_items[neg_item_indices]
    train_loss = bpr_loss(emb_users_final, emb_users, emb_pos_items_final, emb_pos_items, emb_neg_items_final, emb_neg_items)
    train_loss.backward()
    optimizer.step()
    

def main():
    #training()
    movies, ratings, num_users, num_items, user_ids, item_ids, edge_index = getEdgeIndices()
    train_index, train_edge_index, test_edge_index, val_edge_index = getTrainTestValIndices(ratings, edge_index)
    checkpoint = torch.load('model.pt', weights_only=True)
    model = LightGCN(num_users, num_items)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    test_loss, test_recall, test_ndcg = test(model, test_edge_index.to(device), [train_edge_index, val_edge_index])
    print(f"Test loss: {test_loss:.5f} | Test recall@{K}: {test_recall:.5f} | Test ndcg@{K}: {test_ndcg:.5f}")
    #recommendationForNewUser(model, optimizer, 91254, [318, 4776, 76093], [9, 10, 7])

if __name__ == "__main__":
    main()