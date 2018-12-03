import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

#Load the u.user file into a dataframe
u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']

users = pd.read_csv('./data/movielens/u.user', sep='|', names=u_cols,
 encoding='latin-1')

print(users.head())

#Load the u.item file into a dataframe
i_cols = ['movie_id', 'title' ,'release date','video release date', 'IMDb URL', 'unknown', 'Action', 'Adventure',
 'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

movies = pd.read_csv('./data/movielens/u.item', sep='|', names=i_cols, encoding='latin-1')

print(movies.head())
#Remove all information except Movie ID and title
movies = movies[['movie_id', 'title']]

#Load the u.data file into a dataframe
r_cols = ['user_id', 'movie_id', 'rating', 'timestamp']

ratings = pd.read_csv('./data/movielens/u.data', sep='\t', names=r_cols,
 encoding='latin-1')

print(ratings.head())

#Drop the timestamp column
ratings = ratings.drop('timestamp', axis=1)

#Assign X as the original ratings dataframe and y as the user_id column of ratings.
X = ratings.copy()
y = ratings['user_id']

#Split into training and test datasets, stratified along user_id
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, stratify=y, random_state=42)

#Function that computes the root mean squared error (or RMSE)
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

#Define the baseline model to always return 3.
def baseline(user_id, movie_id):
    return 3.0


# Function to compute the RMSE score obtained on the testing set by a model
def score(cf_model):
    # Construct a list of user-movie tuples from the testing dataset
    id_pairs = zip(X_test['user_id'], X_test['movie_id'])

    # Predict the rating for every user-movie tuple
    y_pred = np.array([cf_model(user, movie) for (user, movie) in id_pairs])

    # Extract the actual ratings given by the users in the test data
    y_true = np.array(X_test['rating'])

    # Return the final RMSE score
    return rmse(y_true, y_pred)

print(score(baseline))

