import os
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from typing import Tuple, List
import sys

# ─── CONFIGURATION ────────────────────────────────────────────────────────────
TILE_SIZE = 1024
BATCH_SIZE = 10
NUM_WORKERS = 0
OUT_IMG = Path("tiles/images")
OUT_MSK = Path("tiles/masks")
INDEX_CSV = Path("data/index.csv")
MODEL_PATH = Path("best_model.pth")

# ─── DEVICE SELECTION ────────────────────────────────────────────────────────
def select_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("✅ Using CUDA (GPU)")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")  # for Apple Silicon
        print("🍏 Using Apple MPS (Metal)")
    else:
        device = torch.device("cpu")
        print("🧠 Using CPU")
    return device

# ─── DATASET CLASS ────────────────────────────────────────────────────────
class TiledSegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, images, masks, tile_size, transform=None):
        self.images = images
        self.masks = masks
        self.tile_size = tile_size
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = cv2.imread(str(self.images[idx]), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(self.masks[idx]), cv2.IMREAD_GRAYSCALE)

        if img is None or mask is None:
            raise RuntimeError(f"Failed to load {self.images[idx]} or its mask")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img, mask = augmented['image'], augmented['mask']

        mask = mask.unsqueeze(0)
        mask = mask.float() / 255.0

        return img, mask, str(self.images[idx])  # Return path for identification

# ─── SPLIT TILES BASED ON INDEX.CSV ────────────────────────────────────────
def split_tiles_by_index(img_tiles_dir, msk_tiles_dir, index_csv):
    df = pd.read_csv(index_csv)
    
    # Extract target names for training and validation
    train_targets = df[df['training or validation'] == 'training']['target'].unique()
    val_targets = df[df['training or validation'] == 'validation']['target'].unique()
    
    # Get all tiles
    all_imgs = sorted(img_tiles_dir.glob("*.png"))
    
    train_images = []
    val_images = []
    
    for img_path in all_imgs:
        # Extract original target name (before the double underscore)
        original_target = img_path.stem.split('__')[0]
        
        if original_target in train_targets:
            train_images.append(img_path)
        elif original_target in val_targets:
            val_images.append(img_path)
        else:
            # If not specified, default to training
            train_images.append(img_path)
    
    # Derive masks
    train_masks = [msk_tiles_dir / f"{p.stem}.png" for p in train_images]
    val_masks = [msk_tiles_dir / f"{p.stem}.png" for p in val_images]
    
    return train_images, train_masks, val_images, val_masks

# ─── VISUALIZATION FUNCTIONS ────────────────────────────────────────────────
def denormalize_image(img_normalized: np.ndarray) -> np.ndarray:
    """Convert normalized image back to display range"""
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    img_unnormalized = (img_normalized * std + mean)
    img_unnormalized = np.clip(img_unnormalized, 0, 1)
    return img_unnormalized

def visualize_predictions(model, dataloader, device, num_examples=10, dataset_name=""):
    """Visualize predictions with better image handling"""
    model.eval()
    count = 0
    
    with torch.no_grad():
        for images, masks, paths in dataloader:
            if count >= num_examples:
                break
                
            images = images.to(device)
            outputs = model(images)
            preds = torch.sigmoid(outputs) > 0.5  # binary threshold
            
            for i in range(images.size(0)):
                if count >= num_examples:
                    break
                
                # Get image, mask, prediction
                img = images[i].cpu().permute(1, 2, 0).numpy()
                img = denormalize_image(img)
                
                pred_mask = preds[i].cpu().squeeze().numpy()
                true_mask = masks[i].cpu().squeeze().numpy()
                
                # Extract filename and coordinates from path
                tile_path = Path(paths[i])
                parts = tile_path.stem.split('__')
                target_name = parts[0]
                if len(parts) > 1:
                    coords = parts[1].split('_')
                    row, col = int(coords[0]), int(coords[1])
                    title_suffix = f" - Tile ({row}, {col})"
                else:
                    title_suffix = ""
                
                # Create visualization
                fig, axs = plt.subplots(1, 3, figsize=(15, 5))
                
                axs[0].imshow(img)
                axs[0].set_title(f"Image: {target_name}{title_suffix}")
                
                axs[1].imshow(true_mask, cmap="gray")
                axs[1].set_title("Ground Truth Mask")
                
                axs[2].imshow(pred_mask, cmap="gray")
                axs[2].set_title("Predicted Mask")
                
                for ax in axs:
                    ax.axis("off")
                    
                plt.suptitle(f"{dataset_name} - Example {count + 1}", fontsize=16)
                plt.tight_layout()
                plt.show()
                
                count += 1

def visualize_overlay(model, dataloader, device, num_examples=10, dataset_name=""):
    """Visualize predictions with corrected overlay visualization"""
    model.eval()
    count = 0

    with torch.no_grad():
        for images, masks, paths in dataloader:
            if count >= num_examples:
                break

            images = images.to(device)
            outputs = model(images)
            preds = torch.sigmoid(outputs) > 0.5  # Boolean tensor

            for i in range(images.size(0)):
                if count >= num_examples:
                    break

                # --- Prepare image ---
                img = images[i].cpu().permute(1, 2, 0).numpy()
                img = denormalize_image(img)

                # --- Binarize masks ---
                true_mask = masks[i].cpu().squeeze().numpy()
                gt_bool   = true_mask > 0.5

                pred_bool = preds[i].cpu().squeeze().numpy().astype(bool)

                # --- Build overlays ---
                # 1) Ground truth overlay
                img_gt = img.copy()
                img_gt[gt_bool] = [0, 1, 0]  # green

                # 2) Prediction overlay
                img_pred = img.copy()
                img_pred[pred_bool] = [1, 0, 0]  # red

                # 3) Combined (correct / FP / FN)
                img_all = img.copy()
                correct       = gt_bool & pred_bool
                false_positive= (~gt_bool) & pred_bool
                false_negative= gt_bool & (~pred_bool)

                img_all[correct]        = [0, 1, 0]  # green
                img_all[false_positive] = [1, 0, 0]  # red
                img_all[false_negative] = [0, 0, 1]  # blue

                # 4) Get target name from path
                pth = Path(paths[i])
                nm = pth.stem.split('__')[0]

                # --- Plot ---
                fig, axs = plt.subplots(1, 4, figsize=(20, 5))
                axs[0].imshow(img);       axs[0].set_title(f"Original ({nm.capitalize()})"); 
                axs[1].imshow(img_gt);    axs[1].set_title("Image + GT"); 
                axs[2].imshow(img_pred);  axs[2].set_title("Image + Pred"); 
                axs[3].imshow(img_all);   axs[3].set_title("Combined (G=✓, R=FP, B=FN)")

                for ax in axs:
                    ax.axis("off")

                plt.suptitle(f"{dataset_name} - Overlay Example {count+1}", fontsize=16)
                plt.tight_layout()
                plt.show()

                count += 1

# ─── MAIN EXECUTION ──────────────────────────────────────────────────────
def main():
    device = select_device()
    
    # Check if model exists
    if not MODEL_PATH.exists():
        print(f"Error: Model file not found at {MODEL_PATH}")
        sys.exit(1)
    
    # Check if tiles exist
    if not OUT_IMG.exists() or not OUT_MSK.exists():
        print("Error: Tile directories not found. Please run training script first.")
        sys.exit(1)
    
    # Split tiles based on index.csv
    train_images, train_masks, val_images, val_masks = split_tiles_by_index(OUT_IMG, OUT_MSK, INDEX_CSV)
    
    print(f"Found {len(train_images)} training tiles and {len(val_images)} validation tiles")
    
    # Define transforms (same as in training but without augmentation)
    transform = A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ], additional_targets={'mask': 'mask'})
    
    # Create datasets
    train_dataset = TiledSegmentationDataset(
        images=train_images,
        masks=train_masks,
        tile_size=TILE_SIZE,
        transform=transform
    )
    
    val_dataset = TiledSegmentationDataset(
        images=val_images,
        masks=val_masks,
        tile_size=TILE_SIZE,
        transform=transform
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,  # Shuffle to see variety
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,  # Shuffle to see variety
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    
    # Load model
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,  # Don't load pretrained weights
        in_channels=3,
        classes=1
    ).to(device)
    
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    
    # Visualize predictions
    print("\n=== Training Examples ===")
    visualize_predictions(model, train_loader, device, num_examples=50, dataset_name="Training Set")
    
    print("\n=== Validation Examples ===")
    visualize_predictions(model, val_loader, device, num_examples=50, dataset_name="Validation Set")
    
    # Visualize overlays
    print("\n=== Training Overlay Examples ===")
    visualize_overlay(model, train_loader, device, num_examples=50, dataset_name="Training Set")
    
    print("\n=== Validation Overlay Examples ===")
    visualize_overlay(model, val_loader, device, num_examples=50, dataset_name="Validation Set")

if __name__ == "__main__":
    main()
