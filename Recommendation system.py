#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd

# Load MovieLens dataset
ratings = pd.read_csv(r"C:\Users\Noel\Downloads\Zidio Development Internship\Recommendation System\Netflix_Dataset_Rating.csv")  # userId, movieId, rating
movies = pd.read_csv(r"C:\Users\Noel\Downloads\Zidio Development Internship\Recommendation System\Netflix_Dataset_Movie.csv")    # movieId, title, genres

print(ratings.head())
print(movies.head())


# In[9]:


# Check for missing values
print(ratings.isnull().sum())
print(movies.isnull().sum())

# Normalize ratings (optional)
ratings['rating'] = ratings['rating'] / ratings['rating'].max()


# In[10]:


print(ratings.columns)


# In[11]:


ratings.rename(columns=lambda x: x.strip().lower(), inplace=True)
print(ratings.columns)  # Confirm the renaming


# In[15]:


ratings = pd.read_csv(r"C:\Users\Noel\Downloads\Zidio Development Internship\Recommendation System\Netflix_Dataset_Rating.csv")  # Update with the correct file path
ratings


# In[14]:


import pandas as pd

# Load the dataset
ratings = pd.read_csv(r"C:\Users\Noel\Downloads\Zidio Development Internship\Recommendation System\Netflix_Dataset_Rating.csv")  # Replace with your dataset file path

# Inspect column names
print(ratings.columns)

# Normalize ratings (if 'rating' column exists)
if 'rating' in ratings.columns:
    ratings['rating'] = ratings['rating'] / ratings['rating'].max()
else:
    print("The 'rating' column is missing in the dataset.")


# In[17]:


import pandas as pd

# Load the dataset (replace 'ratings.csv' with your actual file name)
ratings = pd.read_csv(r"C:\Users\Noel\Downloads\Zidio Development Internship\Recommendation System\Netflix_Dataset_Rating.csv")

# Check the first few rows
print(ratings.head())

# Check column names and types
print(ratings.info())

# Check for missing values
print(ratings.isnull().sum())

# Print dataset statistics
print(ratings.describe())


# In[19]:


ratings = ratings.dropna()
ratings['rating'] = ratings['rating'].fillna(ratings['rating'].mean())


# In[21]:


print(ratings.columns)
print(ratings.head())
ratings = pd.read_csv(r"C:\Users\Noel\Downloads\Zidio Development Internship\Recommendation System\Netflix_Dataset_Rating.csv", names=['userID', 'movieID', 'rating', 'timestamp'], header=None)
ratings = ratings.rename(columns={'Rating': 'rating'})


# In[22]:


ratings['rating'] = ratings['rating'].fillna(ratings['rating'].mean())
import pandas as pd

# Load dataset
ratings = pd.read_csv(r"C:\Users\Noel\Downloads\Zidio Development Internship\Recommendation System\Netflix_Dataset_Rating.csv", names=['userID', 'movieID', 'rating', 'timestamp'], header=None)

# Print column names and first few rows to confirm structure
print(ratings.columns)
print(ratings.head())

# Handle missing values
ratings = ratings.dropna()
ratings['rating'] = ratings['rating'].fillna(ratings['rating'].mean())

# Print summary to confirm changes
print(ratings.info())


# In[23]:


ratings = ratings.dropna()
ratings['rating'] = ratings['rating'].fillna(ratings['rating'].mean())


# In[24]:


ratings = ratings.drop_duplicates()
print(f"Number of rows after removing duplicates: {len(ratings)}")


# In[25]:


print(f"Number of unique users: {ratings['userID'].nunique()}")
print(f"Number of unique movies: {ratings['movieID'].nunique()}")


# In[26]:


if 'rating' in ratings.columns:
    ratings['rating'] = ratings['rating'] / ratings['rating'].max()
else:
    print("The 'rating' column is missing.")


# In[27]:


# Pivot table
ratings_matrix = ratings.pivot(index='movieID', columns='userID', values='rating')

# Replace NaN with 0 if needed
ratings_matrix = ratings_matrix.fillna(0)
print(ratings_matrix.head())


# In[29]:


import matplotlib.pyplot as plt

# Distribution of ratings
plt.hist(ratings['rating'], bins=10, edgecolor='black')
plt.title('Distribution of Ratings')
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.show()

# Number of ratings per movie
ratings_per_movie = ratings.groupby('movieID')['rating'].count()
plt.hist(ratings_per_movie, bins=20, edgecolor='black')
plt.title('Number of Ratings per Movie')
plt.xlabel('Number of Ratings')
plt.ylabel('Frequency')
plt.show()


# In[31]:


from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Create a pivot table if not already done
ratings_matrix = ratings.pivot(index='movieID', columns='userID', values='rating').fillna(1)

# Compute cosine similarity
movie_similarity = cosine_similarity(ratings_matrix)
movie_similarity_df = pd.DataFrame(movie_similarity, index=ratings_matrix.index, columns=ratings_matrix.index)

print(movie_similarity_df.head())


# In[32]:


def recommend_movies(movie_id, n=5):
    if movie_id not in movie_similarity_df.index:
        return f"Movie ID {movie_id} not found in the dataset."
    
    # Sort by similarity and exclude the movie itself
    similar_movies = movie_similarity_df[movie_id].sort_values(ascending=False)
    similar_movies = similar_movies.drop(movie_id)  # Exclude the input movie

    return similar_movies.head(n)

# Example: Recommend movies similar to movieID 1
print(recommend_movies(1, n=5))


# In[ ]:




