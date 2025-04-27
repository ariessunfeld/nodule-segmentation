import sys
import os
import re
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def count_white_blobs(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not load image at path: {image_path}")
    _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    num_labels, _ = cv2.connectedComponents(binary, connectivity=8)
    return num_labels - 1

def process_data_folder(data_folder="data"):
    counts = {}
    if not os.path.exists(data_folder):
        raise ValueError(f"Data folder does not exist: {data_folder}")

    for folder_name in os.listdir(data_folder):
        folder_path = os.path.join(data_folder, folder_name)
        if os.path.isdir(folder_path):
            for file_name in os.listdir(folder_path):
                if re.match(r'mask.*\.png$', file_name, re.IGNORECASE):
                    image_path = os.path.join(folder_path, file_name)
                    try:
                        count = count_white_blobs(image_path)
                        counts[folder_name] = count
                    except Exception as e:
                        print(f"{folder_name}: Error processing {file_name} ({str(e)})")
                    break  # Only the first matching file
            else:
                print(f"{folder_name}: No mask*.png file found")
    return counts

def categorize_and_plot(counts_dict):
    # Filter out zero counts
    filtered_counts = {k: v for k, v in counts_dict.items() if v > 0}

    if not filtered_counts:
        print("No non-zero nodule counts to plot.")
        return

    counts = list(filtered_counts.values())
    locations = list(filtered_counts.keys())

    # Calculate thresholds
    low_threshold = np.percentile(counts, 33)
    high_threshold = np.percentile(counts, 66)

    # Categorize each count
    categories = []
    for count in counts:
        if count <= low_threshold:
            categories.append('Low')
        elif count <= high_threshold:
            categories.append('Medium')
        else:
            categories.append('High')

    color_map = {'Low': 'blue', 'Medium': 'orange', 'High': 'red'}
    colors = [color_map[cat] for cat in categories]

    # Sort by count for nicer plot
    sorted_data = sorted(zip(locations, counts, colors), key=lambda x: x[1])
    sorted_locations, sorted_counts, sorted_colors = zip(*sorted_data)

    # Plot
    plt.figure(figsize=(14, 8))
    plt.bar(range(len(sorted_counts)), sorted_counts, color=sorted_colors, edgecolor='black')
    plt.xlabel("Locations (sorted by nodule count)")
    plt.ylabel("Number of Nodules")
    plt.title("Nodule Counts Categorized as Low, Medium, High")
    plt.xticks(ticks=range(len(sorted_locations)), labels=sorted_locations, rotation=90)
    plt.grid(True, axis='y')
    plt.tight_layout()

    legend_handles = [mpatches.Patch(color=color, label=label) for label, color in color_map.items()]
    plt.legend(handles=legend_handles, title="Categories")
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) == 2:
        image_path = sys.argv[1]
        try:
            count = count_white_blobs(image_path)
            print(f"Number of nodules: {count}")
        except Exception as e:
            print(f"Error: {str(e)}")
            sys.exit(1)
    else:
        counts = process_data_folder()
        categorize_and_plot(counts)
