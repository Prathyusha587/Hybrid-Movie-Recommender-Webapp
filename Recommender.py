#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import nltk
import pickle
import os


# In[2]:
print(os.getcwd())

os.chdir(r"C:\Users\91951\Desktop\MRS")



# In[3]:

movies = pd.read_csv(r'C:\Users\91951\Desktop\MRS\datasets\tmdb_5000_movies.csv')
df = pd.read_csv(r'C:\Users\91951\Desktop\MRS\datasets\tmdb_5000_credits.csv')
df.columns = ['id', 'tittle', 'cast', 'crew']
movies = movies.merge(df, on='id')

movies.info(8)


# #### Content based recommendation engine - TF-IDF Vectorizer

# In[4]:


from sklearn.feature_extraction.text import TfidfVectorizer
# removing english stop word like a, and , the 
tfidf = TfidfVectorizer(analyzer = 'word',stop_words = 'english')
#NaN -> ‘’
movies['overview'] = movies['overview'].fillna('')
tfidf_matrix = tfidf.fit_transform(movies['overview'])
tfidf_matrix.shape


# #### Calculation of similarity score by using COSINE SIMILARITY algorithm
# The shape of the TF-IDF matrix is (4803, 20978). which means that here are 20978 different words are used to describe a 4803 movies.
# 
# Now, we will find similarity score of this matrix.
# 
# As we have a TF_IDF vectorizer, calculating directly a dot product will give us a cosine similarity. here we are using cosine similarity score since it is relatively easy and fast to calculate.

# In[5]:


from sklearn.metrics.pairwise import linear_kernel
cosin_sim = linear_kernel(tfidf_matrix, tfidf_matrix)


# Next,we perform reverse map for the indices of the movies and titles.

# In[6]:


index_of_movies = pd.Series(movies.index,   index=movies['title']).drop_duplicates()


# Now, let’s write function for recommendation.
# 
# Step 1: Fetch the title
# Step 2: Compute the similarity score for the movie from cosin_sim matrix
# Step 3: Sort the similarity score
# Step 4: OUTPUT- Return top movie base on the input

# In[7]:


def get_recommendations(title, cosin_sim=cosin_sim):
    idx = index_of_movies[title]
    
    sim_scores = list(enumerate(cosin_sim[idx]))
    # sorting of moviesidx based on similarity score
    sim_scores = sorted(sim_scores, key = lambda x:x[1], reverse = True)
    # get top 10 of sorted 
    sim_scores = sim_scores[1:31]
    
    movies_idx = [i[0] for i in sim_scores]
    
    return movies['title'].iloc[movies_idx]


# In[8]:


get_recommendations('Blood Ties').head(10)


# #### Improvement of recommender with the help of other metadatas

# First, we get the cast, crew, keywords and genres column data. then we will put some preprocessing on that data to get the most useful information for example we will get Director from the ‘crew’ column.
# 
# We will create a soup of these information. and apply the CountVectorizer.
# 
# One important difference is that we use the CountVectorizer() instead of TF-IDF. This is because we do not want to down-weight the presence of an actor/director if he or she has acted or directed in relatively more movies. It doesn’t make much intuitive sense.
# 
# Next step is to compute a Cosine Similarity matrix based on the Count matrix.

# In[9]:


from ast import literal_eval
features = ['cast', 'crew', 'keywords', 'genres']
for f in features:
    movies[f] = movies[f].apply(literal_eval)
# to get director from job
def get_director(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']
    return np.nan
# get top 3 elements of list
def get_list(x):
    if isinstance(x, list):
        names = [ i['name'] for i in x]
        
        if len(names)  > 3:
            names = names[:3]
        return names
    return []
#apply all functions
movies['director'] = movies['crew'].apply(get_director)
features = ['cast', 'keywords', 'genres']
for f in features:
    movies[f] = movies[f].apply(get_list)
#striping
def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(' ', '')) for i in x]
    else:
        if isinstance(x, str):
            return str.lower(x.replace(' ', ''))
        else:
            return ''
features = ['cast', 'keywords', 'director', 'genres']
for f in features:
    movies[f] = movies[f].apply(clean_data)
#creating a SOUP
def create_soup(x):
    return ' '.join(x['keywords'])+' '+' '.join(x['cast'])+' '+x['director']+' '+' '.join(x['genres'])
movies['soup'] = movies.apply(create_soup, axis=1)
#count Vectorizer
from sklearn.feature_extraction.text import CountVectorizer
count = CountVectorizer(stop_words = 'english')
count_matrix = count.fit_transform(movies['soup'])
# finding similarity matrix
from sklearn.metrics.pairwise import cosine_similarity
cosin_sim2 = cosine_similarity(count_matrix, count_matrix)


# In[10]:


get_recommendations('Blood Ties', cosin_sim2)


# ### COLLABORATIVE FILTERING - Python's Surprise library 

# In[11]:


from surprise import Reader, Dataset, SVD
from surprise.model_selection import cross_validate
reader = Reader()
ratings = pd.read_csv(r'C:\Users\91951\Desktop\MRS\datasets\ratings.csv')


# In[12]:


ratings.head(6)


# #### Cross validation of our data

# In[13]:


from surprise import Reader, Dataset, SVD
from surprise.model_selection import cross_validate
reader = Reader(rating_scale=(0,5))



# In[14]:


data = Dataset.load_from_df(ratings[['userId', 'movieId','rating']], reader)
svd = SVD()


# In[15]:


# Run 5 fold cross validation
cross_validate(svd, data,measures=['RMSE'], cv=5,verbose=True)


# We got a Root Mean Square Error of 0.89 approx which is more than good enough for our case. 
# Let us now train on our dataset and arrive at predictions.

# In[16]:


train = data.build_full_trainset()
svd.fit(train)


# Let’s predict the user 1’s rating on the movie Id=302

# In[17]:


svd.predict(1, 302)


# Here , est=2.6280 means that user 1 might  give rating of 2.63 to movie which has Id 302.
# 
# That is how we can predict the movie rating based on the users profile and recommend the best movie to them without knowing the past behaviour of the User. 
# 
# This is called a  collaborative filtering.

# #### HYBRID RECOMMENDER 

# A hybrid model for recommendation of movies to users with the best possible efficiency and precision can be designed as mentioned below:
# 
# 
# Let's put our content based and CF based together and make a strong recommender.

# In[18]:


movie_id = pd.read_csv('../datasets/links_small.csv')

movie_id.dropna()


# In[19]:


movie_id.head(10)


# In[20]:


new_movies = movies.filter(['id', 'title','tagline'])


# In[21]:


new_movies.head(5)


# In[ ]:





# In[22]:


movie_id.head(5)


# In[23]:


# convert float val to int
def conv_int(x):
    try:
        return int(x)
    except:
        return np.nan


# In[24]:


movie_id = pd.read_csv('../datasets/links_small.csv')[['movieId', 'tmdbId']]
movie_id['tmdbId'] = movie_id['tmdbId'].apply(conv_int)
movie_id.columns = ['movieId', 'id']
movie_id = movie_id.merge(new_movies[['title', 'id']], on='id').set_index('title')
print(movie_id.shape)
movie_id


# Next, we make a index_map to find a index of a movie.

# In[25]:


index_map = movie_id.set_index('id')




# #### Count Vectorization

# In[26]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 5000, stop_words='english')
count_matrix = cv.fit_transform(movies['soup']).toarray()


# In[27]:


count_matrix


# In[28]:


from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity(count_matrix)


# In[29]:


similarity


# In[30]:


len(similarity[0])


# In[31]:


sorted(list(enumerate(similarity[0])), reverse=True, key = lambda x:x[1])


# ####  Below function is the main function i.e.., hybrid recommender function which has the power of two recommendation techniques(Content-based filtering and Collaborative filtering) combined

# In[32]:


def recommend_for(userid, title):
   index = index_of_movies[title]
   tmdbId = movie_id.loc[title]['id']
   
   
   
    #content based
   sim_scores = list(enumerate(cosin_sim2[int(index)]))
   sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
   sim_scores = sim_scores[1:10]
   movie_indices = [i[0] for i in sim_scores]

   mv = movies.iloc[movie_indices][['title', 'vote_count', 'vote_average', 'id']]
   mv = mv[mv['id'].isin(movie_id['id'])]
    
   
  
   #collaborative filtering - svd
   

   mv['est'] = mv['id'].apply(lambda x: svd.predict(userid, index_map.loc[x]['movieId']).est)

   mv = mv.sort_values('est', ascending=False)
   
   
   
   return mv.head(10)


# In[33]:


recommend_for(343, "Avatar")


# In[34]:


import pickle
pickle.dump(new_movies, open('../model/movies_list.pkl', 'wb'))
pickle.dump(similarity, open('../model/similarity.pkl', 'wb'))

