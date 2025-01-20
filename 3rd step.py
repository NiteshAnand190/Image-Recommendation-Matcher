import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import pickle
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

# Load the embeddings
image_embeddings = np.load('image_embeddings.npy')
text_embeddings = np.load('text_embeddings.npy')

# Load the dataframe for image details
csv_file = './new_style.csv'
df = pd.read_csv(csv_file)

# Load the saved tfidf_vectorizer using pickle
with open('tfidf_vectorizer.pkl', 'rb') as file:
    tfidf_vectorizer = pickle.load(file)

# Dimensionality reduction to match the text embeddings' size
pca = PCA(n_components=text_embeddings.shape[1])  # Reduce to 1344 components
reduced_image_embeddings = pca.fit_transform(image_embeddings)

# Normalize the reduced image embeddings and text embeddings
reduced_image_embeddings = normalize(reduced_image_embeddings)
text_embeddings = normalize(text_embeddings)

# Function to compute similarity between a query (text or image) and all image embeddings
def find_similar_images(query_embed, image_embeddings, top_k=5):
    similarities = cosine_similarity(query_embed, image_embeddings)
    top_k_indices = similarities.argsort()[0][-top_k:][::-1]  # Reverse to get top k most similar
    return top_k_indices

# Function for text query to find similar images
def find_similar_images_from_text(query_text, top_k=5):
    query_text_embed = tfidf_vectorizer.transform([query_text]).toarray()
    query_text_embed = normalize(query_text_embed)
    similar_image_indices = find_similar_images(query_text_embed, reduced_image_embeddings, top_k)
    return df.iloc[similar_image_indices]

# Function for image query to find similar images
def find_similar_images_from_image(image_id, top_k=5):
    image_idx = np.where(df['id'].values == image_id)[0][0]
    query_image_embed = reduced_image_embeddings[image_idx].reshape(1, -1)
    similar_image_indices = find_similar_images(query_image_embed, reduced_image_embeddings, top_k)
    return df.iloc[similar_image_indices]

# Example: Find the most similar images to a text query (e.g., "blue shirt")
query_text = "blue shirt"
similar_images_from_text = find_similar_images_from_text(query_text)
print("Similar images from text query:")
print(similar_images_from_text[['id', 'productDisplayName']])

# Example: Find the most similar images to an image query (e.g., using image ID)
query_image_id = 3446  # Replace with the ID of the image you want to query
similar_images_from_image = find_similar_images_from_image(query_image_id)
print(f"\nSimilar images from image ID {query_image_id}:")
print(similar_images_from_image[['id', 'productDisplayName']])
