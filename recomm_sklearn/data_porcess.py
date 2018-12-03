import pandas as pd
import numpy as np

df = pd.read_csv('./data/movies_metadata.csv')
df = df[['title','genres', 'release_date', 'runtime', 'vote_average', 'vote_count']]

print df.head()

#Convert release_date into pandas datetime format
df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')

#Extract year from the datetime
df['year'] = df['release_date'].apply(lambda x: str(x).split('-')[0] if x != np.nan else np.nan)

#Helper function to convert NaT to 0 and all other years to integers.
def convert_int(x):
    try:
        return int(x)
    except:
        return 0
#Apply convert_int to the year feature
df['year'] = df['year'].apply(convert_int)

#Drop the release_date column
df = df.drop('release_date', axis=1)

#Display the dataframe
print df.head()

#Print genres of the first movie
print df.iloc[0]['genres']


#Import the literal_eval function from ast
from ast import literal_eval

#Define a stringified list and output its type
a = "[1,2,3]"
print(type(a))

#Apply literal_eval and output type
b = literal_eval(a)
print(type(b))

#Convert all NaN into stringified empty lists
df['genres'] = df['genres'].fillna('[]')

#Apply literal_eval to convert stringified empty lists to the list object
df['genres'] = df['genres'].apply(literal_eval)

#Convert list of dictionaries to a list of strings
df['genres'] = df['genres'].apply(lambda x: [i['name'].lower() for i in x] if isinstance(x, list) else [])
print df.head()


#Create a new feature by exploding genres
s = df.apply(lambda x: pd.Series(x['genres']),axis=1).stack().reset_index(level=1, drop=True)

#Name the new feature as 'genre'
s.name = 'genre'

#Create a new dataframe gen_df which by dropping the old 'genres' feature and adding the new 'genre'.
gen_df = df.drop('genres', axis=1).join(s)

#Print the head of the new gen_df
print gen_df.head()


def build_chart(gen_df, percentile=0.8):
    # Ask for preferred genres
    print("Input preferred genre")
    genre = input()

    # Ask for lower limit of duration
    print("Input shortest duration")
    low_time = int(input())

    # Ask for upper limit of duration
    print("Input longest duration")
    high_time = int(input())

    # Ask for lower limit of timeline
    print("Input earliest year")
    low_year = int(input())

    # Ask for upper limit of timeline
    print("Input latest year")
    high_year = int(input())

    # Define a new movies variable to store the preferred movies. Copy the contents of gen_df to movies
    movies = gen_df.copy()

    # Filter based on the condition
    movies = movies[(movies['genre'] == genre) &
                    (movies['runtime'] >= low_time) &
                    (movies['runtime'] <= high_time) &
                    (movies['year'] >= low_year) &
                    (movies['year'] <= high_year)]

    # Compute the values of C and m for the filtered movies
    C = movies['vote_average'].mean()
    m = movies['vote_count'].quantile(percentile)

    # Only consider movies that have higher than m votes. Save this in a new dataframe q_movies
    q_movies = movies.copy().loc[movies['vote_count'] >= m]

    # Calculate score using the IMDB formula
    q_movies['score'] = q_movies.apply(lambda x: (x['vote_count'] / (x['vote_count'] + m) * x['vote_average'])
                                                 + (m / (m + x['vote_count']) * C)
                                       , axis=1)

    # Sort movies in descending order of their scores
    q_movies = q_movies.sort_values('score', ascending=False)

    return q_movies

print build_chart(gen_df).head()

df.to_csv('./data/metadata_clean.csv', index=False)

# Input preferred genre
#
# action
#
# Input shortest duration
#
# 80
#
# Input longest duration
#
# 120
#
# Input earliest year
#
# 1990
#
# Input latest year
#
# 2000


