from __future__ import print_function
#description-based recommender
import pandas as pd
import numpy as np
# Load the keywords and credits files
cred_df = pd.read_csv('./data/credits.csv')
key_df = pd.read_csv('./data/keywords.csv')

df = pd.read_csv('./data/metadata_clean.csv')
#Import the original file
orig_df = pd.read_csv('./data/movies_metadata.csv', low_memory=False)

#Add the useful features into the cleaned dataframe
df['overview'], df['id'] = orig_df['overview'], orig_df['id']

print(cred_df.head())
print(cred_df.columns)
def clean_ids(x):
    try:
        return int(x)
    except:
        return np.nan
#Clean the ids of df
df['id'] = df['id'].apply(clean_ids)

#Filter all rows that have a null ID
df = df[df['id'].notnull()]
# Convert IDs into integer
df['id'] = df['id'].astype('int')
key_df['id'] = key_df['id'].astype('int')
cred_df['id'] = cred_df['id'].astype('int')

# Merge keywords and credits into your main metadata dataframe
df = df.merge(cred_df, on='id')
df = df.merge(key_df, on='id')

#Display the head of df
print(df.head())


# Convert the stringified objects into the native python objects
from ast import literal_eval

features = ['cast', 'crew', 'keywords', 'genres']
for feature in features:
    df[feature] = df[feature].apply(literal_eval)

#Print the first cast member of the first movie in df
print(df.iloc[0]['crew'][0])

def get_director(x):
    for crew_member in x:
        if crew_member['job'] == 'Director':
            return crew_member['name']
    return np.nan
#Define the new director feature
df['director'] = df['crew'].apply(get_director)

#Print the directors of the first five movies
print(df['director'].head())

# Returns the list top 3 elements or entire list; whichever is more.
def generate_list(x):
    if isinstance(x, list):
        names = [i['name'] for i in x]
        #Check if more than 3 elements exist. If yes, return only first three. If no, return entire list.
        if len(names) > 3:
            names = names[:3]
        return names

    #Return empty list in case of missing/malformed data
    return []
#Apply the generate_list function to cast and keywords
df['cast'] = df['cast'].apply(generate_list)
df['keywords'] = df['keywords'].apply(generate_list)
#Only consider a maximum of 3 genres
df['genres'] = df['genres'].apply(lambda x: x[:3])
# Print the new features of the first 5 movies along with title
print(df[['title', 'cast', 'director', 'keywords', 'genres']].head())

# Function to sanitize data to prevent ambiguity. It removes spaces and converts to lowercase
def sanitize(x):
    if isinstance(x, list):
        #Strip spaces and convert to lowercase
        return [str.lower(i.replace(" ", "")) for i in x]
    else:
        #Check if director exists. If not, return empty string
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        else:
            return ''
for feature in ['cast', 'director', 'genres', 'keywords']:
    df[feature] = df[feature].apply(sanitize)
#Function that creates a soup out of the desired metadata
def create_soup(x):
    s = ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + x['director'] + ' ' + ' '.join(x['genres'])
    #s = ' '.join(x['keywords']) + ' ' +.join(x['cast']) + ' ' + x['director'] + ' ' + ' '.join(x['genres'])
    #s = s.encode('utf-8');
    return s

# Create the new soup feature
df['soup'] = df.apply(create_soup, axis=1)

print(df.iloc[0]['soup'])

# Import CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer

#Define a new CountVectorizer object and create vectors for the soup
count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(df['soup'])
#Import cosine_similarity function
from sklearn.metrics.pairwise import cosine_similarity

#Compute the cosine similarity score (equivalent to dot product for tf-idf vectors)
cosine_sim = cosine_similarity(count_matrix, count_matrix)
# Reset index of your df and construct reverse mapping again
df = df.reset_index()
indices = pd.Series(df.index, index=df['title'])
def content_recommender(title, cosine_sim=cosine_sim, df=df, indices=indices):
    # Obtain the index of the movie that matches the title
    idx = indices[title]

    # Get the pairwsie similarity scores of all movies with that movie
    # And convert it into a list of tuples as described above
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the cosine similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies. Ignore the first movie.
    sim_scores = sim_scores[1:11]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]
    print(movie_indices)

    # Return the top 10 most similar movies
    return df['title'].iloc[movie_indices]
movie_recomm = content_recommender('The Lion King', cosine_sim, df, indices)
print(movie_recomm)


