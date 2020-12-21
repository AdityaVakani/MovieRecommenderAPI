from flask import Flask, request, jsonify
from flask_mysqldb import MySQL
from model import recommend_top_all,recommend_top_genre,recommend_similar
import numpy as np
import json

app = Flask(__name__)

app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'password'
app.config['MYSQL_DB'] = 'tmdb'

mysql = MySQL(app)

@app.route('/search/<title>', methods=['GET'])
def search(title):

        cur = mysql.connect.cursor()
        try:
            cur.execute("SELECT title from titles where title like %s or title like %s limit 10", (title +"%","% "+title +"%"))
        except:
            print("cant execute sql")
        response = cur.fetchall()
        items =[]
        for row in response:
            for key in cur.description:
                items.append({key[0]:value for value in row})
        res_json = json.dumps(items)
        # mysql.connection.commit()
        return res_json



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