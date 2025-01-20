import pandas as pd
import shutil
import os
import requests

# Google Drive file IDs for CSV and Images
csv_file_url = 'https://drive.google.com/uc?id=1o2S3KktAvFdcgD9y5s-aCDlMXbWDw1PA'  # Corrected URL for CSV file
image_folder_id = '1R1-aXAhSKAHdR0XCVcqZy5-GwN00F4JZ'  # Folder ID for images, but each image file needs to be accessed via its ID

# Load CSV directly from Google Drive
df = pd.read_csv(csv_file_url)

# Define folders
images_folder = './images'
new_images_folder = './new_images'
new_csv_file = './new_style.csv'

# Process as usual
df_sorted = df.sort_values(by='id')
df_subset = df_sorted.head(2000)

# Create folder for new images
os.makedirs(new_images_folder, exist_ok=True)

# For each image, download using Google Drive direct download link
for image_id in df_subset['id']:
    image_filename = f"{image_id}.jpg"  # Assuming image extension is .jpg, adjust if necessary
    image_url = f'https://drive.google.com/uc?id={image_id}'  # Create a direct download URL using the image ID

    # Download image
    response = requests.get(image_url)
    if response.status_code == 200:
        with open(os.path.join(new_images_folder, image_filename), 'wb') as f:
            f.write(response.content)

# Save the new CSV file
df_subset.to_csv(new_csv_file, index=False)

print("Extraction and saving completed.")
