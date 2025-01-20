import pandas as pd
import numpy as np
import os
import requests
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# Google Drive URLs for CSV and Images
csv_file_url = 'https://drive.google.com/file/d/1o2S3KktAvFdcgD9y5s-aCDlMXbWDw1PA/view?usp=drive_link'  # Replace with your actual CSV file ID
image_folder_id = '1R1-aXAhSKAHdR0XCVcqZy5-GwN00F4JZ'  # Replace with your folder ID or individual image file IDs


# Function to download files from Google Drive
def download_file_from_drive(url, dest_path):
    response = requests.get(url)
    if response.status_code == 200:
        with open(dest_path, 'wb') as f:
            f.write(response.content)
    else:
        print(f"Failed to download {url}")


# Download CSV file
csv_file = './new_style.csv'
download_file_from_drive(csv_file_url, csv_file)

# Load the CSV file
df = pd.read_csv(csv_file)


# Load and preprocess images
def preprocess_images(image_folder, image_ids):
    images = []
    for img_id in image_ids:
        # Download image from Google Drive using its file ID
        image_url = f'https://drive.google.com/uc?id={image_folder_id}/{img_id}.jpg'  # Update the image file path accordingly
        img_path = os.path.join(image_folder, f"{img_id}.jpg")

        # Download each image to the local directory
        download_file_from_drive(image_url, img_path)

        # Load the image using Keras
        img = image.load_img(img_path, target_size=(224, 224))  # Resize to 224x224 for ResNet
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        images.append(img)
    return np.vstack(images)


# Get the image IDs from the dataframe
image_ids = df['id'].values
image_folder = './new_images'

# Preprocess the images
images = preprocess_images(image_folder, image_ids)

# Load pre-trained ResNet model to extract features
resnet_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

# Extract image features (embeddings) from ResNet
image_embeddings = resnet_model.predict(images)

# Preprocess the textual data
# Convert the textual features into embeddings using TF-IDF
text_features = df[['gender', 'masterCategory', 'subCategory', 'articleType', 'baseColour', 'season', 'usage',
                    'productDisplayName']].astype(str).agg(' '.join, axis=1)

tfidf_vectorizer = TfidfVectorizer(max_features=5000)
text_embeddings = tfidf_vectorizer.fit_transform(text_features).toarray()

# Normalize both the image and text embeddings
from sklearn.preprocessing import normalize

image_embeddings = normalize(image_embeddings)
text_embeddings = normalize(text_embeddings)

print("Image embeddings shape:", image_embeddings.shape)
print("Text embeddings shape:", text_embeddings.shape)

# Save the image embeddings
np.save('image_embeddings.npy', image_embeddings)

# Save the text embeddings
np.save('text_embeddings.npy', text_embeddings)

print("Embeddings saved successfully!")

# Save the tfidf_vectorizer using pickle
with open('tfidf_vectorizer.pkl', 'wb') as file:
    pickle.dump(tfidf_vectorizer, file)

print("TF-IDF Vectorizer saved successfully!")
