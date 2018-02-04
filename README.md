# Recommendation System 
Recommendation engines usually produce a set of recommendations using either collaborative filtering or
content-based filtering. The difference between the two approaches is in the way the recommendations
are mined. Collaborative filtering builds a model from the past behavior of the current user as well as
ratings given by other users. We then use this model to predict what this user might be interested in.
Content-based filtering, on the other hand, uses the characteristics of the item itself in order to recommend
more items to the user. The similarity between items is the main driving force here.
# Find users that are similar
One of the most important tasks in building a recommendation engine is finding users that are similar.
# Find similar users
It takes three input arguments: the
database, input user, and the number of similar users that we are looking for. Our first step is to
check whether the user is present in the database. If the user exists, we need to compute the Pearson
correlation score between this user and all the other users in the database.
