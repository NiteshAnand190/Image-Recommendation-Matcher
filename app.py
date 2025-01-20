import streamlit as st
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import pickle
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import ResNet50
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from tensorflow.keras.applications.resnet50 import preprocess_input

# Load embeddings
image_embeddings = np.load('image_embeddings.npy')
text_embeddings = np.load('text_embeddings.npy')

# Load dataframe
csv_file = './new_style.csv'
df = pd.read_csv(csv_file)

# Load TF-IDF vectorizer
with open('tfidf_vectorizer.pkl', 'rb') as file:
    tfidf_vectorizer = pickle.load(file)

# Dimensionality reduction to match the image embeddings' size
pca = PCA(n_components=1344)  # Reduce to 1344 components
reduced_image_embeddings = pca.fit_transform(image_embeddings)

# Dimensionality reduction for text embeddings
pca_text = PCA(n_components=1344)  # Reduce to 1344 components (same as image embeddings)
reduced_text_embeddings = pca_text.fit_transform(text_embeddings)

# Normalize embeddings
reduced_image_embeddings = normalize(reduced_image_embeddings)
reduced_text_embeddings = normalize(reduced_text_embeddings)

# Load ResNet50 model
resnet_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

# Function to find similar images
def find_similar_images(query_embed, image_embeddings, top_k=5):
    similarities = cosine_similarity(query_embed, image_embeddings)
    top_k_indices = similarities.argsort()[0][-top_k:]
    return top_k_indices


# Streamlit UI
st.title("Image and Text Search for Fashion Products")

option = st.selectbox("Search by", ["Text Query", "Image Upload"])

if option == "Text Query":
    query_text = st.text_input("Enter your search text", "")
    if query_text:
        # Step 1: Get the text embedding using TF-IDF
        query_text_embed = tfidf_vectorizer.transform([query_text]).toarray()

        # Step 2: Normalize the text embedding
        query_text_embed = normalize(query_text_embed)

        # Step 3: Compute cosine similarity between the query and image embeddings
        # Find similar images
        similar_image_indices = find_similar_images(query_text_embed, reduced_image_embeddings)

        # Step 4: Retrieve similar images from the dataframe
        similar_images = df.iloc[similar_image_indices]

        # Step 5: Display the results
        st.write("Similar Images:")
        for idx in similar_images['id']:
            st.image(f"C:/Users/nites/Desktop/ASSEMBLE/new images/{idx}.jpg", caption=f"ID: {idx}",
                     use_column_width=True)

elif option == "Image Upload":
    uploaded_image = st.file_uploader("Choose an image...", type="jpg")
    if uploaded_image:
        img = image.load_img(uploaded_image, target_size=(224, 224))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        img_embed = resnet_model.predict(img)
        img_embed = normalize(img_embed)

        # Apply PCA to reduce the dimensionality of the uploaded image embedding
        img_embed_reduced = pca.transform(img_embed)

        similar_image_indices = find_similar_images(img_embed_reduced, reduced_image_embeddings)
        similar_images = df.iloc[similar_image_indices]

        st.write("Similar Images:")
        for idx in similar_images['id']:
            st.image(f"./new images/{idx}.jpg", caption=f"ID: {idx}",
                     use_column_width=True)

