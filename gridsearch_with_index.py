import os
from pathlib import Path
import shutil
import numpy as np
from PIL import Image
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import time
import cv2
from itertools import product


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


# ─── SET SEED ───────────────────────────────────────────────────────────────
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if torch.backends.mps.is_available():
        torch.manual_seed(seed)


# ─── MASK VERIFICATION HELPER ───────────────────────────────────────────────
def verify_mask(arr: np.ndarray) -> np.ndarray:
    if arr.ndim != 2 or not np.array_equal(np.unique(arr), [0, 255]):
        gray = Image.fromarray(arr).convert("L")
        a = np.array(gray)
        return (a >= 255).astype(np.uint8) * 255
    return arr


# ─── TILING FUNCTION ───────────────────────────────────────────────────────
def create_tiles(img_dir, msk_dir, out_img, out_msk, tile_size):
    # Wipe existing tiles to avoid contamination
    if out_img.exists():
        shutil.rmtree(out_img)
    if out_msk.exists():
        shutil.rmtree(out_msk)
    
    out_img.mkdir(parents=True, exist_ok=True)
    out_msk.mkdir(parents=True, exist_ok=True)
    
    for img_path in sorted(img_dir.glob("*.tif")):
        stem = img_path.stem
        mask_path = msk_dir / f"{stem}.png"
        if not mask_path.exists():
            print(f"⚠️ no mask for {stem}, skipping")
            continue

        img_arr = np.array(Image.open(img_path))
        msk_arr = verify_mask(np.array(Image.open(mask_path)))

        h, w = msk_arr.shape[:2]
        for i in range(0, h, tile_size):
            for j in range(0, w, tile_size):
                name = f"{stem}__{i:05d}_{j:05d}"  # Using double underscore and padding for cleaner splits

                tile_img = img_arr[i:i+tile_size, j:j+tile_size]
                tile_msk = msk_arr[i:i+tile_size, j:j+tile_size]

                if tile_img.dtype != np.uint8:
                    tile_img8 = (tile_img >> 8).astype(np.uint8)
                else:
                    tile_img8 = tile_img

                if tile_img8.ndim == 2:
                    tile_img8 = np.stack([tile_img8]*3, axis=-1)

                Image.fromarray(tile_img8).save(out_img / f"{name}.png")
                Image.fromarray(tile_msk, mode="L").save(out_msk / f"{name}.png")

    print("✅ Tiling complete!")


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
    
    print(f"→ {len(train_images)} training tiles, {len(val_images)} validation tiles")
    
    # Verify all masks exist
    missing = [p for p in train_masks + val_masks if not p.exists()]
    if missing:
        print(f"⚠️ Missing mask files for: {missing}")
    
    return train_images, train_masks, val_images, val_masks


# ─── DATASET CLASS ────────────────────────────────────────────────────────
class TiledSegmentationDataset(Dataset):
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

        return img, mask


# ─── COMBINED LOSS ────────────────────────────────────────────────────────
class CombinedLoss(nn.Module):
    def __init__(self, dice_weight=1.0, bce_weight=1.0):
        super().__init__()
        self.dice = smp.losses.DiceLoss(mode='binary')
        self.bce = nn.BCEWithLogitsLoss()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight

    def forward(self, outputs, targets):
        return self.dice_weight * self.dice(outputs, targets) + self.bce_weight * self.bce(outputs, targets)


# ─── TRAINING FUNCTION ────────────────────────────────────────────────────
def train_one_epoch(model, dataloader, optimizer, loss_fn, device):
    model.train()
    running_loss = 0.0
    total_iou = 0.0
    total_dice = 0.0
    total_accuracy = 0.0
    num_batches = 0

    for images, masks in tqdm(dataloader, desc="Training"):
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        with torch.no_grad():
            preds = torch.sigmoid(outputs) > 0.5
            preds = preds.float()

            intersection = (preds * masks).sum(dim=(1, 2, 3))
            union = ((preds + masks) > 0).sum(dim=(1, 2, 3))
            iou = (intersection / (union + 1e-7)).mean().item()

            dice = (2 * intersection / (preds.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3)) + 1e-7)).mean().item()
            accuracy = (preds == masks).float().mean().item()

            total_iou += iou
            total_dice += dice
            total_accuracy += accuracy
            num_batches += 1

    return {
        "loss": running_loss / num_batches,
        "iou": total_iou / num_batches,
        "dice": total_dice / num_batches,
        "accuracy": total_accuracy / num_batches
    }


# ─── VALIDATION FUNCTION ─────────────────────────────────────────────────
def validate(model, dataloader, loss_fn, device):
    model.eval()
    val_loss = 0.0
    total_iou = 0.0
    total_dice = 0.0
    total_accuracy = 0.0
    num_batches = 0

    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="Validation"):
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = loss_fn(outputs, masks)
            val_loss += loss.item()

            preds = torch.sigmoid(outputs) > 0.5
            preds = preds.float()

            intersection = (preds * masks).sum(dim=(1, 2, 3))
            union = ((preds + masks) > 0).sum(dim=(1, 2, 3))
            iou = (intersection / (union + 1e-7)).mean().item()

            dice = (2 * intersection / (preds.sum(dim=(1,2,3)) + masks.sum(dim=(1,2,3)) + 1e-7)).mean().item()
            accuracy = (preds == masks).float().mean().item()

            total_iou += iou
            total_dice += dice
            total_accuracy += accuracy
            num_batches += 1

    return {
        "loss": val_loss / num_batches,
        "iou": total_iou / num_batches,
        "dice": total_dice / num_batches,
        "accuracy": total_accuracy / num_batches
    }


# ─── CONSOLIDATED TRAINING FUNCTION FOR GRID SEARCH ─────────────────────────
def train_model_with_hyperparams(
    img_dir=Path("data/tifs"), 
    msk_dir=Path("data/handcrafted_masks"),
    index_csv=Path("data/index.csv"),
    tile_size=512,
    batch_size=4,
    learning_rate=5e-4,
    num_epochs=50,
    optimizer_name="adam",
    dice_weight=1.0,
    bce_weight=1.0,
    seed=100,
    num_workers=0
):
    """
    Train a model with the given hyperparameters and return the best validation IoU
    along with the hyperparameters used.
    
    Returns:
        dict: A dictionary containing the best validation IoU and the hyperparameters used
    """
    # Setup
    device = select_device()
    set_seed(seed)
    
    # Create temporary directories for tiles
    out_img = Path(f"tiles_{tile_size}/images")
    out_msk = Path(f"tiles_{tile_size}/masks")
    
    # Create tiles - wiped and redone each time
    create_tiles(img_dir, msk_dir, out_img, out_msk, tile_size)
    
    # Split tiles based on index.csv
    train_images, train_masks, val_images, val_masks = split_tiles_by_index(out_img, out_msk, index_csv)
    
    # Define transforms
    train_transform = A.Compose([
        A.HorizontalFlip(p=0.4),
        A.RandomRotate90(p=0.4),
        A.RandomBrightnessContrast(p=0.4),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ], additional_targets={'mask': 'mask'})
    
    val_transform = A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ], additional_targets={'mask': 'mask'})
    
    # Create datasets
    train_dataset = TiledSegmentationDataset(
        images=train_images,
        masks=train_masks,
        tile_size=tile_size,
        transform=train_transform
    )
    
    val_dataset = TiledSegmentationDataset(
        images=val_images,
        masks=val_masks,
        tile_size=tile_size,
        transform=val_transform
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Initialize model
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1
    ).to(device)
    
    # Initialize loss function with weights
    loss_fn = CombinedLoss(dice_weight=dice_weight, bce_weight=bce_weight)
    
    # Initialize optimizer based on the specified name
    if optimizer_name.lower() == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_name.lower() == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    elif optimizer_name.lower() == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    else:
        print(f"Unknown optimizer: {optimizer_name}, defaulting to Adam")
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    best_iou = 0.0
    train_losses, val_losses = [], []
    train_ious, val_ious = [], []
    
    print(f"Starting training with: Tile Size={tile_size}, Batch Size={batch_size}, LR={learning_rate}, Optimizer={optimizer_name}")
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        
        train_metrics = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_metrics = validate(model, val_loader, loss_fn, device)
        
        # Store metrics
        train_losses.append(train_metrics['loss'])
        val_losses.append(val_metrics['loss'])
        train_ious.append(train_metrics['iou'])
        val_ious.append(val_metrics['iou'])
        
        # Print metrics
        print(f"Train Loss: {train_metrics['loss']:.4f} | "
              f"Train IoU: {train_metrics['iou']:.4f} | "
              f"Train Dice: {train_metrics['dice']:.4f} | "
              f"Train Acc: {train_metrics['accuracy']:.4f} || "
              f"Val Loss: {val_metrics['loss']:.4f} | "
              f"Val IoU: {val_metrics['iou']:.4f} | "
              f"Val Dice: {val_metrics['dice']:.4f} | "
              f"Val Acc: {val_metrics['accuracy']:.4f}")
        
        # Update best IoU
        if val_metrics['iou'] > best_iou:
            best_iou = val_metrics['iou']
            
        print(f"Epoch {epoch + 1}/{num_epochs}, Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
    
    # Clean up tile directories
    if out_img.exists():
        shutil.rmtree(out_img)
    if out_msk.exists():
        shutil.rmtree(out_msk)
    
    # Return the best validation IoU and the hyperparameters used
    results = {
        "best_val_iou": best_iou,
        "hyperparameters": {
            "tile_size": tile_size,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "optimizer_name": optimizer_name,
            "dice_weight": dice_weight,
            "bce_weight": bce_weight
        },
        "training_history": {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "train_ious": train_ious,
            "val_ious": val_ious
        }
    }
    
    return results


# ─── GRID SEARCH FUNCTION ────────────────────────────────────────────────
def grid_search(
    tile_sizes=[256, 512, 768],
    batch_sizes=[2, 4, 8],
    learning_rates=[1e-4, 5e-4, 1e-3],
    optimizers=["adam", "adamw"],
    dice_weights=[1.0],
    bce_weights=[1.0],
    num_epochs=15,  # Reduced for grid search
    seed=100
):
    """
    Perform grid search over the given hyperparameters.
    
    Returns:
        dict: The best results from the grid search
    """
    best_result = {
        "best_val_iou": 0.0,
        "hyperparameters": None
    }
    
    all_results = []
    
    # Create all combinations of hyperparameters
    param_combinations = list(product(
        tile_sizes, 
        batch_sizes, 
        learning_rates, 
        optimizers,
        dice_weights,
        bce_weights
    ))
    
    print(f"Starting grid search with {len(param_combinations)} parameter combinations")
    
    for i, (tile_size, batch_size, lr, optimizer_name, dice_weight, bce_weight) in enumerate(param_combinations):
        print(f"\n======= Grid Search: {i+1}/{len(param_combinations)} =======")
        print(f"Parameters: Tile Size={tile_size}, Batch Size={batch_size}, LR={lr}, "
              f"Optimizer={optimizer_name}, Dice Weight={dice_weight}, BCE Weight={bce_weight}")
        
        # Train the model with the current hyperparameters
        result = train_model_with_hyperparams(
            tile_size=tile_size,
            batch_size=batch_size,
            learning_rate=lr,
            optimizer_name=optimizer_name,
            dice_weight=dice_weight,
            bce_weight=bce_weight,
            num_epochs=num_epochs,
            seed=seed
        )
        
        all_results.append(result)
        
        # Update best result if better IoU is found
        if result["best_val_iou"] > best_result["best_val_iou"]:
            best_result = {
                "best_val_iou": result["best_val_iou"],
                "hyperparameters": result["hyperparameters"]
            }
            
            print(f"✅ New best IoU: {best_result['best_val_iou']:.4f} with hyperparameters:")
            for key, value in best_result["hyperparameters"].items():
                print(f"  - {key}: {value}")
    
    # Save all results to CSV for analysis
    results_df = []
    for result in all_results:
        row = {"best_val_iou": result["best_val_iou"]}
        row.update(result["hyperparameters"])
        results_df.append(row)
        
    results_df = pd.DataFrame(results_df)
    results_df.to_csv("grid_search_results.csv", index=False)
    print(f"Grid search results saved to grid_search_results.csv")
    
    return best_result


# ─── MAIN EXECUTION ──────────────────────────────────────────────────────
if __name__ == "__main__":
    best_params = grid_search(
        tile_sizes=[128, 256, 512, 1024],
        batch_sizes=[4, 8, 12],
        learning_rates=[1e-5, 5e-5, 1e-4, 2e-4, 4e-4],
        optimizers=["adam", "adamw", "sgd"],
        num_epochs=20
    )
    
    print("\n=== Best Parameters Found ===")
    print(f"Best Validation IoU: {best_params['best_val_iou']:.4f}")
    for key, value in best_params["hyperparameters"].items():
        print(f"{key}: {value}")
