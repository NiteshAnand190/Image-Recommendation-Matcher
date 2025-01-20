import pandas as pd
import shutil
import os

# Paths
csv_file = 'C:/Users/nites/Desktop/ASSEMBLE/styles.csv'
images_folder = 'C:/Users/nites/Desktop/ASSEMBLE/images'
new_images_folder = 'C:/Users/nites/Desktop/ASSEMBLE/new images'
new_csv_file = 'C:/Users/nites/Desktop/ASSEMBLE/new_style.csv'

# Step 1: Load the CSV file
df = pd.read_csv(csv_file)

# Step 2: Sort the CSV based on the 'id' column
df_sorted = df.sort_values(by='id')

# Step 3: Extract the first 2000 rows after sorting
df_subset = df_sorted.head(2000)

# Step 4: Create a new folder for the first 2000 images
os.makedirs(new_images_folder, exist_ok=True)

# Step 5: Copy the corresponding images to the new folder
for image_id in df_subset['id']:  # Assuming 'id' is the column for image numbering
    image_filename = f"{image_id}.jpg"  # Modify if your images have a different file extension
    image_path = os.path.join(images_folder, image_filename)
    if os.path.exists(image_path):
        shutil.copy(image_path, os.path.join(new_images_folder, image_filename))

# Step 6: Save the new CSV with the first 2000 rows
df_subset.to_csv(new_csv_file, index=False)

print("Extraction and saving completed.")
