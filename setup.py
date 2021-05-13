import numpy as np
import pandas as pd
import warnings


warnings.filterwarnings('ignore')


columns_names = ["user_id","item_id","rating","timestamp"]
df = pd.read_csv("ml-100k/u.data",sep='\t',names=columns_names)


movies_titles = pd.read_csv('ml-100k/u.item',sep="|",header=None)

movies_titles = movies_titles[[0,1]]
movies_titles.columns = ['item_id','title']

df = pd.merge(df,movies_titles,on="item_id")




ratings = pd.DataFrame(df.groupby('title').mean()['rating'])
ratings['num of ratings'] = pd.DataFrame(df.groupby('title').count()['rating'])


moviemat = df.pivot_table(index="user_id", columns ="title", values="rating")



def predict_movies(movie_name):
    movie_user_ratings = moviemat[movie_name]
    similar_to_movie = moviemat.corrwith(movie_user_ratings)
    
    corr_movie = pd.DataFrame(similar_to_movie,columns = ['Corelations'])
    corr_movie.dropna(inplace=True)
    corr_movie = corr_movie.join(ratings['num of ratings'])
    predictions = corr_movie[corr_movie['num of ratings'] >100].sort_values("Corelations", ascending=False)

    return predictions
    


predictions = predict_movies('Goofy Movie, A (1995)')
response = predictions.head(n=20)
response = list(response.index)
for movie in response:
    print(movie)
    # Oooh Big Moives!
