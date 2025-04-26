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

# ─── CONFIGURATION ────────────────────────────────────────────────────────────
# These can be easily changed
TILE_SIZE = 1024
BATCH_SIZE = 4
LEARNING_RATE = 5e-4
NUM_EPOCHS = 80
SEED = 100
NUM_WORKERS = 0
IMG_DIR = Path("data/tifs")
MSK_DIR = Path("data/handcrafted_masks")
OUT_IMG = Path("tiles/images")
OUT_MSK = Path("tiles/masks")
INDEX_CSV = Path("data/index.csv")

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
    def __init__(self):
        super().__init__()
        self.dice = smp.losses.DiceLoss(mode='binary')
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, outputs, targets):
        return self.dice(outputs, targets) + self.bce(outputs, targets)

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

# ─── MAIN EXECUTION ──────────────────────────────────────────────────────
def main():
    # Set up device and seed
    device = select_device()
    set_seed(SEED)
    
    # Create tiles - wiped and redone each time
    create_tiles(IMG_DIR, MSK_DIR, OUT_IMG, OUT_MSK, TILE_SIZE)
    
    # Split tiles based on index.csv
    train_images, train_masks, val_images, val_masks = split_tiles_by_index(OUT_IMG, OUT_MSK, INDEX_CSV)
    
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
        tile_size=TILE_SIZE,
        transform=train_transform
    )
    
    val_dataset = TiledSegmentationDataset(
        images=val_images,
        masks=val_masks,
        tile_size=TILE_SIZE,
        transform=val_transform
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    
    # Initialize model
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1
    ).to(device)
    
    loss_fn = CombinedLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Training loop
    best_iou = 0.0
    train_losses, val_losses = [], []
    train_ious, val_ious = [], []
    train_dices, val_dices = [], []
    train_accuracies, val_accuracies = [], []
    
    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
        
        train_metrics = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_metrics = validate(model, val_loader, loss_fn, device)
        
        # Store metrics
        train_losses.append(train_metrics['loss'])
        val_losses.append(val_metrics['loss'])
        train_ious.append(train_metrics['iou'])
        val_ious.append(val_metrics['iou'])
        train_dices.append(train_metrics['dice'])
        val_dices.append(val_metrics['dice'])
        train_accuracies.append(train_metrics['accuracy'])
        val_accuracies.append(val_metrics['accuracy'])
        
        # Print metrics
        print(f"Train Loss: {train_metrics['loss']:.4f} | "
              f"Train IoU: {train_metrics['iou']:.4f} | "
              f"Train Dice: {train_metrics['dice']:.4f} | "
              f"Train Acc: {train_metrics['accuracy']:.4f} || "
              f"Val Loss: {val_metrics['loss']:.4f} | "
              f"Val IoU: {val_metrics['iou']:.4f} | "
              f"Val Dice: {val_metrics['dice']:.4f} | "
              f"Val Acc: {val_metrics['accuracy']:.4f}")
        
        # Save best model
        if val_metrics['iou'] > best_iou:
            best_iou = val_metrics['iou']
            t0 = time.time()
            torch.save(model.state_dict(), "best_model.pth")
            print(f"✅ Saved best model (based on IoU) [took {time.time() - t0:.2f} seconds]")
        
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}, Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
    
    # Plot results
    plt.figure(figsize=(12, 10))
    
    plt.subplot(2, 2, 1)
    plt.plot(range(1, NUM_EPOCHS + 1), train_losses, label='Train Loss', color='blue')
    plt.plot(range(1, NUM_EPOCHS + 1), val_losses, label='Validation Loss', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.legend()
    
    plt.subplot(2, 2, 2)
    plt.plot(range(1, NUM_EPOCHS + 1), train_ious, label='Train IoU', color='blue')
    plt.plot(range(1, NUM_EPOCHS + 1), val_ious, label='Validation IoU', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('IoU')
    plt.title('IoU over Epochs')
    plt.legend()
    
    plt.subplot(2, 2, 3)
    plt.plot(range(1, NUM_EPOCHS + 1), train_dices, label='Train Dice', color='blue')
    plt.plot(range(1, NUM_EPOCHS + 1), val_dices, label='Validation Dice', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Dice Coefficient')
    plt.title('Dice Coefficient over Epochs')
    plt.legend()
    
    plt.subplot(2, 2, 4)
    plt.plot(range(1, NUM_EPOCHS + 1), train_accuracies, label='Train Accuracy', color='blue')
    plt.plot(range(1, NUM_EPOCHS + 1), val_accuracies, label='Validation Accuracy', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy over Epochs')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
