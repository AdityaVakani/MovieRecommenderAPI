from flask import Flask,request
from model import recommend_top_all,recommend_top_genre,recommend_similar
import numpy as np

app = Flask(__name__)


@app.route('/top',defaults={'no_items':None},methods=['GET'])
@app.route('/top/<no_items>',methods=['GET'])
def top(no_items):
    if no_items == None:
        return recommend_top_all()
    return recommend_top_all(int(no_items))


@app.route('/genre/<genre>',defaults ={'no_items':None},methods=['GET'])
@app.route('/genre/<genre>/<no_items>',methods=['GET'])
def top_genre(genre,no_items):
    #genre=request.args.get(genre)
    try:
        if no_items == None:
            return recommend_top_genre(genre)
        else:
            return recommend_top_genre(genre,int(no_items))
    except:
        return "genre not found"


@app.route('/similar/<movie>',defaults={'no_items':None},methods=['GET'])
@app.route('/similar/<movie>/<no_items>',methods=['GET'])
def similar_movies(movie,no_items):
    try:
        if no_items == None:
            return recommend_similar(movie)
        else:
            return recommend_similar(movie,int(no_items))
    except:
        return "Movie Not Found"

if __name__ == '__main__':
    app.run(debug=True)