from flask import Flask, request, url_for, redirect, render_template
from flask_cors import CORS
import pandas as pd
import pickle
import torch
import os
app = Flask(__name__)
CORS(app)
#model = pickle.load(open("../entire_model.pt", "rb"))
model = torch.load("../entire_model.pt", weights_only=False)


@app.route("/")
def use_template():
    return render_template("index.html")

@app.route("/predict", methods=['POST'])
def predict():
    app.logger.info(request.form)
    input_one = request.form["1"]
    input_two = request.form["2"]
    user_id = int(input_one)
    num_recs = int(input_two)

    movie_path = '../../Recommender System/Dataset/ml-latest-small/movies.csv'
    rating_path = '../../Recommender System/Dataset/ml-latest-small/ratings.csv'
    movies = pd.read_csv(movie_path)
    ratings = pd.read_csv(rating_path)
    ratings = ratings.loc[ratings['movieId'].isin(movies['movieId'].unique())]

    user_mapping = {userid: i for i, userid in enumerate(ratings['userId'].unique())}
    item_mapping = {isbn: i for i, isbn in enumerate(ratings['movieId'].unique())}
    user_ids = torch.LongTensor([user_mapping[i] for i in ratings['userId']])
    item_ids = torch.LongTensor([item_mapping[i] for i in ratings['movieId']])
    edge_index = torch.stack((user_ids, item_ids))
    user_pos_items = dict()
    for i in range(edge_index.shape[1]):
        user = edge_index[0][i].item()
        item = edge_index[1][i].item()
        if user not in user_pos_items:
            user_pos_items[user] = []
        user_pos_items[user].append(item)
    app.logger.info(user_mapping)
    if user_id not in user_mapping:
        return render_template('usernotfound.html')
    user = user_mapping[user_id]
    #emb_user = model.emb_users.weight[user]
    emb_user = model['emb_users.weight'][user]
    #ratings = model.emb_items.weight @ emb_user
    ratings = model['emb_items.weight'] @ emb_user

    movie_title = pd.Series(movies['title'].values, index=movies.movieId).to_dict()

    values, indices = torch.topk(ratings, k=100)

    ids = [index.cpu().item() for index in indices if index in user_pos_items[user]][:num_recs]
    movies = [list(item_mapping.keys())[list(item_mapping.values()).index(movie)] for movie in ids]
    titles = [movie_title[id] for id in movies]

    print(f'Favorite movies from user n°{user_id}:')
    for i in range(len(movies)):
        print(f'- {titles[i]}')

    ids = [index.cpu().item() for index in indices if index not in user_pos_items[user]][:num_recs]
    movies = [list(item_mapping.keys())[list(item_mapping.values()).index(movie)] for movie in ids]
    titles = [movie_title[id] for id in movies]

    print(f'\nRecommended movies for user n°{user_id}')
    for i in range(num_recs):
        print(f'- {titles[i]}')
    
    #return render_template('result.html', pred = titles)
    return render_template('result.html', pred=titles)

if __name__ == "__main__":
    app.run(debug=True)