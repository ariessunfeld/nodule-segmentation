import sys
import os
import re
import cv2
import numpy as np

def count_white_blobs(image_path):
    # Load image in grayscale mode
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError(f"Could not load image at path: {image_path}")

    # Threshold the image: convert to binary (0 or 255)
    _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    # Find connected components
    # connectivity=8 considers diagonal pixels as connected (for blobs)
    num_labels, _ = cv2.connectedComponents(binary, connectivity=8)

    # Subtract 1 to remove the background label
    num_blobs = num_labels - 1

    return num_blobs

def process_data_folder(data_folder="data"):
    if not os.path.exists(data_folder):
        raise ValueError(f"Data folder does not exist: {data_folder}")

    for folder_name in sorted(os.listdir(data_folder)):
        folder_path = os.path.join(data_folder, folder_name)
        if os.path.isdir(folder_path):
            # Find the first file matching mask*.png
            for file_name in os.listdir(folder_path):
                if re.match(r'mask.*\.png$', file_name, re.IGNORECASE):
                    image_path = os.path.join(folder_path, file_name)
                    try:
                        count = count_white_blobs(image_path)
                        print(f"{folder_name}: {count} nodules")
                    except Exception as e:
                        print(f"{folder_name}: Error processing {file_name} ({str(e)})")
                    break  # Only process the first matching image
            else:
                print(f"{folder_name}: No mask*.png file found")

if __name__ == "__main__":
    if len(sys.argv) == 2:
        image_path = sys.argv[1]
        try:
            count = count_white_blobs(image_path)
            print(f"Number of nodules: {count}")
        except Exception as e:
            print(f"Error: {str(e)}")
            sys.exit(1)
    elif len(sys.argv) == 1:
        process_data_folder()
    else:
        print("Usage: python count_nodules.py <image_path>")
        sys.exit(1)
