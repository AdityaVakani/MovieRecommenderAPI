
import pandas as pd
import numpy as np
from scipy import stats,sparse
from ast import literal_eval
from flask import jsonify
# import pickle
import joblib
import json


import warnings; warnings.simplefilter('ignore')


def prepare_data(data_location):
    md = pd. read_csv(data_location)
    md['genres'] = md['genres'].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
    md['year'] = pd.to_datetime(md['release_date'], errors='coerce').apply(lambda x: str(x).split('-')[0] if x != np.nan else np.nan)
    vote_counts = md[md['vote_count'].notnull()]['vote_count'].astype('int')
    vote_averages = md[md['vote_average'].notnull()]['vote_average'].astype('int')
    C = vote_averages.mean()
    m = vote_counts.quantile(0.95)
    print("in  prepare data ")
    return md ,vote_counts,vote_averages,C,m

print("calling prepare data")
md,vote_counts,vote_averages,C,m = prepare_data('./lib/data/tmdb/movies_metadata.csv')



def weighted_rating(x):
    #print("In weighted rating")
    # vote_counts = md[md['vote_count'].notnull()]['vote_count'].astype('int')
    # vote_averages = md[md['vote_average'].notnull()]['vote_average'].astype('int')
    # C = vote_averages.mean()
    # m = vote_counts.quantile(0.95)
    v = x['vote_count']
    R = x['vote_average']
    return (v/(v+m) * R) + (m/(m+v) * C)


def prepare_top_all():
    print("preparing top all")
    # vote_counts = md[md['vote_count'].notnull()]['vote_count'].astype('int')
    # vote_averages = md[md['vote_average'].notnull()]['vote_average'].astype('int')
    # C = vote_averages.mean()
    # m = vote_counts.quantile(0.95)

    qualified = md[(md['vote_count'] >= m) & (md['vote_count'].notnull()) & (md['vote_average'].notnull())][['title', 'year', 'vote_count', 'vote_average', 'popularity', 'genres','poster_path']]
    qualified['vote_count'] = qualified['vote_count'].astype('int')
    qualified['vote_average'] = qualified['vote_average'].astype('int')

    qualified['wr'] = qualified.apply(weighted_rating, axis=1)
    qualified = qualified.sort_values('wr', ascending=False).head(250)
    return qualified

print("calling preparing top all")
top_all = prepare_top_all()


def recommend_top_all(no_items=15):
    print("in recommend top all")
    return json.loads(top_all.head(no_items).to_json(orient="table"))


def prepare_rec_genre():
    print("in prepare rec genre")
    s = md.apply(lambda x: pd.Series(x['genres']),axis=1).stack().reset_index(level=1, drop=True)
    s.name = 'genre'
    gen_md = md.drop('genres', axis=1).join(s)
    return gen_md

print("calling prepare rec genre")
gen_md=prepare_rec_genre()



def recommend_top_genre(genre,no_items=15,percentile=0.85):
    print("In recommend top_genre")
    df = gen_md[gen_md['genre'] == genre]
    vote_counts = df[df['vote_count'].notnull()]['vote_count'].astype('int')
    vote_averages = df[df['vote_average'].notnull()]['vote_average'].astype('int')
    C = vote_averages.mean()
    m = vote_counts.quantile(percentile)

    qualified = df[(df['vote_count'] >= m) & (df['vote_count'].notnull()) & (df['vote_average'].notnull())][
        ['title', 'year', 'vote_count', 'vote_average', 'popularity','poster_path']]
    qualified['vote_count'] = qualified['vote_count'].astype('int')
    qualified['vote_average'] = qualified['vote_average'].astype('int')

    qualified['wr'] = qualified.apply(
        lambda x: (x['vote_count'] / (x['vote_count'] + m) * x['vote_average']) + (m / (m + x['vote_count']) * C),
        axis=1)
    qualified = qualified.sort_values('wr', ascending=False).head(250)

    return json.loads(qualified.head(no_items).to_json(orient="table"))


def prepare_rec_sim(location):
    print("in prepare rec similar")
    smd = joblib.load(location+'smd')
    scosine_sim0 = joblib.load(location+'scosine_sim0')

    smd = smd.reset_index()
    titles = smd['title']
    indices = pd.Series(smd.index, index=smd['title'])
    return smd,scosine_sim0,titles,indices

print("calling prepare rec similar")
smd,scosine_sim0,titles,indices = prepare_rec_sim('./lib/models/')


def sparse_to_list(idx):

    scx=sparse.coo_matrix(scosine_sim0[idx])
    scx_list=[]
    for i,j,v in zip(scx.row,scx.col,scx.data):
        scx_list.append([j,v])
    return scx_list

def get_recommendations(title):
    idx = indices[title]
    sim_scores=sparse_to_list(idx)
    #sim_scores = list(enumerate(scosine_sim0[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:31]
    movie_indices = [i[0] for i in sim_scores]
    return titles.iloc[movie_indices]


def recommend_similar(title,no_items=10):
    print("in recommend similar")
    idx = indices[title]
    sim_scores = sparse_to_list(idx)
    # sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:]
    movie_indices = [i[0] for i in sim_scores]

    movies = smd.iloc[movie_indices][['id','title', 'vote_count', 'vote_average', 'year','genres','poster_path']]
    vote_counts = movies[movies['vote_count'].notnull()]['vote_count'].astype('int')
    vote_averages = movies[movies['vote_average'].notnull()]['vote_average'].astype('int')
    C = vote_averages.mean()
    m = vote_counts.quantile(0.60)
    qualified = movies[
        (movies['vote_count'] >= m) & (movies['vote_count'].notnull()) & (movies['vote_average'].notnull())]
    qualified['vote_count'] = qualified['vote_count'].astype('int')
    qualified['vote_average'] = qualified['vote_average'].astype('int')
    qualified['wr'] = qualified.apply(weighted_rating, axis=1)
    qualified = qualified.sort_values('wr', ascending=False).head(no_items)
    return json.loads(qualified.head(no_items).to_json(orient="table"))
    #return jsonify(qualified.head(no_items).to_json(orient="table"))

