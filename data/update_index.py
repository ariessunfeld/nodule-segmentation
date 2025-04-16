from pathlib import Path
import os
import pandas as pd

root_dir = Path(__file__).parent
index_csv_path = os.path.join(root_dir, 'index.csv')

# Load existing index
df = pd.read_csv(index_csv_path)

# Function to get image-mask pairs from a target folder
def get_image_mask_pairs(folder_path):
    images = [f for f in os.listdir(folder_path) if f.endswith('.tif')]
    pairs = []
    for img in images:
        base = os.path.splitext(img)[0]
        mask_name = f'mask_{base}.png'
        mask_path = os.path.join(folder_path, mask_name)
        img_path = os.path.join(folder_path, img)
        if os.path.exists(mask_path):
            pairs.append((img_path, mask_path))
    return pairs

# Traverse folders
for target_folder in os.listdir(root_dir):
    folder_path = os.path.join(root_dir, target_folder)
    if os.path.isdir(folder_path) and target_folder not in ['__pycache__']:
        pairs = get_image_mask_pairs(folder_path)
        for img_path, mask_path in pairs:
            new_entry = {
                'target': target_folder,
                'image path': img_path,
                'mask path': mask_path,
                'training or validation': ''  # Leave empty
            }
            df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)

# Drop duplicates if any, based on image path
df.drop_duplicates(subset='image path', inplace=True)

# Save updated index
df.to_csv(index_csv_path, index=False)
