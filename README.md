# nodule-segmentation

Description: TODO

# Getting Started

- Clone the repository
- Install `poetry` if it is not already installed
- Navigate into the repository
- Run the command `poetry install` to install the packages 

# Scripts overview

### `reorganize_data_folders.py`

This script is essential. After adding images and masks to `data/` (following the convention of adding a new folder `data/NewFolder/` and placing the image in the folder as `filename.tif` and the mask as `mask_filename.png`), run this script to copy files into `data/tifs/` and `data/handcrafted_masks/` with appropriate renaming, automatically.

### `data/update_index.py`

After adding new images to `data/`, run this script to update `data/index.csv` with the new information. **Note**: If you have removed folders from `data/`, you will need to delete the relevant rows from `data/index.csv`. They will not be removed automatically. This script assumes that `data/index.csv` exists and has the columns `target,image path,mask path,training or validation`

### `train.py`

This is a training script. It reads in `data/index.csv` to understand which images are for training versus for validation. Specify parameters at the top: learning rate, tile size (must be power of two and less than or equal to 1024), batch size (be careful not to overload your system's memory), etc.

### `visualize.py`

After training, you can run this script to visualize the performance of the best model (saved while training to `best_model.pth`) on both the training images, as well as the validation imges. Edit the parameters at the bottom of the script to control how many images should be shown from each set.

### `gridsearch_with_index.py`

This script performs a gridsearch across learning rates, batch sizes, tile sizes, and optimizers, then saves the results to a `CSV` file. **Note**: Be careful to back up the CSV before re-running the gridsearch to avoid losing the results from the previous run.




# Running scripts (`.py`)

- Navigate into the repository
- Run the command `poetry run python path/to/script.py`

# Running notebooks (`.ipynb`)

## With VSCode

- Open VSCode
- Go to File > Open Folder
- Select this repository
- Press Cmd+Shift+P (Ctrl+Shift+P on Windows)
- Select the python interpreter associated with the Poetry-managed environment (likely the suggested interpreter)
- Open a notebook file (`.ipynb`) or create a new one
- Run the code in the notebook as needed

## Without VSCode

- Run the command `poetry run jupyter notebook`
- Open a notebook file or create a new one in the Browser Jupyter Notebook interface
- Run the code in the notebook as needed
