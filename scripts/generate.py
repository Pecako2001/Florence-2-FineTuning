import os
import json
import pandas as pd
from PIL import Image
import io

# Function to convert non-serializable data to serializable format
def convert_to_serializable(data):
    if isinstance(data, (list, dict)):
        return data
    else:
        return str(data)

# Function to save entries to files
def save_entries_to_files(data, folder_name):
    # Create the folder if it doesn't exist
    os.makedirs(folder_name, exist_ok=True)

    # Iterate through the data and save each entry
    for idx, entry in enumerate(data):
        # Save JSON data
        json_data = {k: convert_to_serializable(v) for k, v in entry.items() if k != 'image'}
        question_id = json_data.get('questionId', idx)
        json_file_path = os.path.join(folder_name, f'{question_id}.json')
        with open(json_file_path, 'w') as f:
            json.dump(json_data, f, indent=4)
        
        # Save image data if exists
        if 'image' in entry:
            image_data_dict = entry['image']
            # Assuming the actual image data is stored under the key 'bytes'
            if 'bytes' in image_data_dict:
                image_data = image_data_dict['bytes']
                image = Image.open(io.BytesIO(image_data)).convert('RGB')
                image_file_path = os.path.join(folder_name, f'{question_id}.png')
                image.save(image_file_path)
            else:
                print(f"No image bytes found for entry {question_id}")

# Load dataset from Parquet file
def load_parquet(parquet_file_path):
    df = pd.read_parquet(parquet_file_path)
    return df.to_dict(orient='records')

# Adjust these paths to your train and validation Parquet files
train_parquet_path = 'data/train-00000-of-00038.parquet'

# Load data
train_data = load_parquet(train_parquet_path)

# Save the entries to files inside the 'loaded_train' and 'loaded_val' folders
save_entries_to_files(train_data, 'dataset')
