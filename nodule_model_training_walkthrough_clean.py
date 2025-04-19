

import matplotlib.pyplot as plt


# sanity check

import sys
import torch

print("Torch version: ", "\n", torch.__version__)
#print("\n")
#print("Torch path: ", "\n",torch.__file__)
print("\n")
print("Python version: ", "\n",sys.version)
#print("\n")
#print("Python path: ", "\n",sys.executable)

import os
from pathlib import Path
print("\n")
#print("Current path: ", "\n",os.getcwd())

# Automatically select device: CUDA > MPS (Apple Silicon) > CPU


if torch.cuda.is_available():
    device = torch.device("cuda")
    print("✅ Using CUDA (GPU)")
elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")  # for Apple Silicon
    print("🍏 Using Apple MPS (Metal)")
else:
    device = torch.device("cpu")
    print("🧠 Using CPU")

# set seeds

import random
import numpy as np

def set_seed(seed: int):
    # Set seed for Python random module
    random.seed(seed)
    
    # Set seed for NumPy
    np.random.seed(seed)
    
    # Set seed for PyTorch (for both CPU and GPU)
    torch.manual_seed(seed)
    
    # If you're using a GPU (CUDA), set the seed for CUDA as well
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups

    # Set seed for Apple Silicon (Metal) GPU if available
    if torch.backends.mps.is_available():  # Check if the MPS backend (Metal) is available
        torch.manual_seed(seed)




# set the seed :)
set_seed(100)

# is the code in a notebook?

USE_NOTEBOOK_LOGIC = True

path_prefix = Path.cwd() 

if not USE_NOTEBOOK_LOGIC:
     path_prefix = Path(__file__).parent 
    

#!/usr/bin/env python3
from pathlib import Path
import shutil
import numpy as np
from PIL import Image

TILE    = 256
IMG_DIR = Path("data/tifs")
MSK_DIR = Path("data/handcrafted_masks")
OUT_IMG = Path("tiles/images")
OUT_MSK = Path("tiles/masks")

# ─── WIPE EXISTING TILES ───────────────────────────────────────────────────────

if OUT_IMG.exists():
    shutil.rmtree(OUT_IMG)
if OUT_MSK.exists():
    shutil.rmtree(OUT_MSK)

OUT_IMG.mkdir(parents=True, exist_ok=True)
OUT_MSK.mkdir(parents=True, exist_ok=True)


# ─── MASK VERIFICATION HELPER ─────────────────────────────────────────────────

def verify_mask(arr: np.ndarray) -> np.ndarray:
    """
    Ensure mask is single-channel binary (0 or 255).
    If not, convert to grayscale then threshold.
    """
    if arr.ndim != 2 or not np.array_equal(np.unique(arr), [0, 255]):
        gray = Image.fromarray(arr).convert("L")
        a = np.array(gray)
        return (a >= 255).astype(np.uint8) * 255
    return arr


# ─── MAIN TILING LOOP ─────────────────────────────────────────────────────────

for img_path in sorted(IMG_DIR.glob("*.tif")):
    stem = img_path.stem
    mask_path = MSK_DIR / f"{stem}.png"
    if not mask_path.exists():
        print(f"⚠️ no mask for {stem}, skipping")
        continue

    # Load full images/masks into memory
    img_arr = np.array(Image.open(img_path))
    msk_arr = verify_mask(np.array(Image.open(mask_path)))

    h, w = msk_arr.shape[:2]
    for i in range(0, h, TILE):
        for j in range(0, w, TILE):
            name = f"{stem}_{i//TILE:02d}_{j//TILE:02d}"

            # extract the tiles
            tile_img = img_arr[i:i+TILE, j:j+TILE]
            tile_msk = msk_arr[i:i+TILE, j:j+TILE]

            # ─── IMAGE: convert 16‑bit→8‑bit by right‑shifting 8 bits, then save PNG
            if tile_img.dtype != np.uint8:
                tile_img8 = (tile_img >> 8).astype(np.uint8)
            else:
                tile_img8 = tile_img

            # if grayscale TIFF, tile_img8 is H×W; convert to RGB
            if tile_img8.ndim == 2:
                tile_img8 = np.stack([tile_img8]*3, axis=-1)

            Image.fromarray(tile_img8).save(OUT_IMG / f"{name}.png")

            # ─── MASK: already 8‑bit binary, save PNG
            Image.fromarray(tile_msk, mode="L").save(OUT_MSK / f"{name}.png")

print("✅ Tiling complete!")


# split into training and validation
# 
from pathlib import Path

# Paths to your tiled folders
IMG_TILES = Path("tiles/images")
MSK_TILES = Path("tiles/masks")

# 1) Grab all image tile paths
all_imgs = sorted(IMG_TILES.glob("*.png"))
print(len(all_imgs))
# 2) Split by stem
val_images = [p for p in all_imgs if "achnacarry" in p.stem.lower() or "ayre_of_tonga" in p.stem.lower() or "addiewell" in p.stem.lower()]
train_images = [p for p in all_imgs if "achnacarry" not in p.stem.lower() and "ayre_of_tonga" not in p.stem.lower() and "addiewell" not in p.stem.lower()]

# 3) Derive masks by swapping folders & extensions
val_masks   = [MSK_TILES / f"{p.stem}.png" for p in val_images]
train_masks = [MSK_TILES / f"{p.stem}.png" for p in train_images]

# 4) Sanity checks
print(f"→ {len(train_images)} training tiles, {len(val_images)} validation tiles")
# Optional: verify every mask file actually exists
missing = [p for p in val_masks + train_masks if not p.exists()]
if missing:
    print("⚠️ Missing mask files for:", missing)


#dataset class
# 

import numpy as np
import torch
import cv2
from torch.utils.data import Dataset

class TiledSegmentationDataset(Dataset):
    def __init__(self, images, masks, tile_size, transform=None):
        self.images    = images
        self.masks     = masks
        self.tile_size = tile_size
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # 1) Load raw tile (keep full bit‑depth)
        #img  = cv2.imread(self.images[idx], cv2.IMREAD_UNCHANGED)
        # 1) Load 3‑channel PNG tile directly
        img  = cv2.imread(self.images[idx], cv2.IMREAD_COLOR)  # always H×W×3 BGR
        mask = cv2.imread(self.masks[idx],  cv2.IMREAD_GRAYSCALE)

        if img is None or mask is None:
            raise RuntimeError(f"Failed to load {self.images[idx]} or its mask")

        # 2) Fake‑RGB for single‑channel TIFFs
        # 2) Convert BGR→RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # if img.ndim == 2:
        #     # H×W → H×W×3 by stacking the gray band into each channel
        #     img = np.stack([img, img, img], axis=-1)
        # else:
        #     # if it is already 3‑channel, convert BGR→RGB
        #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 3) Apply your Albumentations (Normalize + ToTensorV2)
        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img, mask = augmented['image'], augmented['mask']

        # 4) Ensure mask has channel dim
        mask = mask.unsqueeze(0)   # 1×H×W

        #    # 1) add the channel dim to mask
        #mask = mask.unsqueeze(0)  # → 1×H×W
        #mask = mask.float()      # now dtype=torch.float32

        # 2) cast to float and normalize to [0,1]
        mask = mask.float() / 255.0


        return img, mask


#dataloaders

from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

batch_size = 16

# ─── 1) Transforms ──────────────────────────────────────────────────────────────

train_transform = A.Compose([
    A.HorizontalFlip(p=0.4),
    A.RandomRotate90(p=0.4),
    A.RandomBrightnessContrast(p=0.4),
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
], additional_targets={'mask': 'mask'})

val_transform = A.Compose([
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
], additional_targets={'mask': 'mask'})

# ─── 2) Dataset instances ──────────────────────────────────────────────────────

train_dataset = TiledSegmentationDataset(
    images=train_images,
    masks=train_masks,
    tile_size=TILE,
    transform=train_transform
)

val_dataset = TiledSegmentationDataset(
    images=val_images,
    masks=val_masks,
    tile_size=TILE,
    transform=val_transform
)



print(f"→ Total image tiles: {len(train_images) + len(val_images)}")
print(f"→ Train tiles:        {len(train_images)}")
print(f"→ Val tiles:          {len(val_images)}")
# ─── 3) DataLoader instances ───────────────────────────────────────────────────

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,        # or os.cpu_count()
    pin_memory=True,
    drop_last=False        # drop the last partial batch if want all-equal sizes
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)





import segmentation_models_pytorch as smp

model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1).to(device)

#set_seed(21)
best_iou = 0.0

set_seed(100)

print(smp.__version__)


# Loss, Optimizer, Training and Validation Logic

import torch.nn as nn
from tqdm import tqdm

# === Loss function (binary segmentation) ===
# Define combined loss manually
class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.dice = smp.losses.DiceLoss(mode='binary')
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, outputs, targets):
        return self.dice(outputs, targets) + self.bce(outputs, targets)

loss_fn = CombinedLoss()

# === Optimizer ===
optimizer = torch.optim.Adam(model.parameters(), lr=4e-5)

# === Training function ===
def train_one_epoch(model, dataloader, optimizer, loss_fn):
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

        # === Metrics ===
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



# === Validation function ===
def validate(model, dataloader, loss_fn):
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

            # === Metrics ===
            preds = torch.sigmoid(outputs) > 0.5  # threshold at 0.5
            preds = preds.float()

            intersection = (preds * masks).sum(dim=(1, 2, 3))
            union = ((preds + masks) > 0).sum(dim=(1, 2, 3))
            iou = (intersection / (union + 1e-7)).mean().item()

            dice = (2 * intersection / (preds.sum(dim=(1,2,3)) + masks.sum(dim=(1,2,3)) + 1e-7)).mean().item()

            correct = (preds == masks).float().mean().item()

            total_iou += iou
            total_dice += dice
            total_accuracy += correct
            num_batches += 1

    return {
        "loss": val_loss / num_batches,
        "iou": total_iou / num_batches,
        "dice": total_dice / num_batches,
        "accuracy": total_accuracy / num_batches
    }




import time

# model.load_state_dict(torch.load(path_prefix.parent / "scripts" / "best_model.pth", map_location=device))
# model.to(device)
# loss_fn =CombinedLoss()
#optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
#uncomment the lines above to start training over


# === Training loop with Visualization ===

num_epochs = 60

# Lists to store metrics for plotting
train_losses, val_losses = [], []
train_ious, val_ious = [], []
train_dices, val_dices = [], []
train_accuracies, val_accuracies = [], []

for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")

    # Train and validate the model
    train_metrics = train_one_epoch(model, train_loader, optimizer, loss_fn)

    #scheduler.step()
    
    val_metrics = validate(model, val_loader, loss_fn)

    # Store metrics for plotting
    train_losses.append(train_metrics['loss'])
    val_losses.append(val_metrics['loss'])
    train_ious.append(train_metrics['iou'])
    val_ious.append(val_metrics['iou'])
    train_dices.append(train_metrics['dice'])
    val_dices.append(val_metrics['dice'])
    train_accuracies.append(train_metrics['accuracy'])
    val_accuracies.append(val_metrics['accuracy'])

    # Print metrics for this epoch
    print(f"Train Loss: {train_metrics['loss']:.4f} | "
          f"Train IoU: {train_metrics['iou']:.4f} | "
          f"Train Dice: {train_metrics['dice']:.4f} | "
          f"Train Acc: {train_metrics['accuracy']:.4f} || "
          f"Val Loss: {val_metrics['loss']:.4f} | "
          f"Val IoU: {val_metrics['iou']:.4f} | "
          f"Val Dice: {val_metrics['dice']:.4f} | "
          f"Val Acc: {val_metrics['accuracy']:.4f}")

    # Save best model based on IoU
    if val_metrics['iou'] > best_iou:
        best_iou = val_metrics['iou']
        #save_best_full_model(model, optimizer, epoch, val_metrics, "best_full_model.pth", scheduler=None)
        t0 = time.time()
        torch.save(model.state_dict(), "best_model.pth")
        print("✅ Saved best model (based on IoU) [took {:.2f} seconds]".format(time.time() - t0))

    # Update learning rate
    #scheduler.step()  # Update the learning rate according to the schedule

    # print the current learning rate at each epoch
    print(f"Epoch {epoch + 1}/{num_epochs}, Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")    

# After training, plot the metrics
plt.figure(figsize=(12, 10))

# Plot Loss
plt.subplot(2, 2, 1)
plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss', color='blue')
plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss', color='red')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss over Epochs')
plt.legend()

# Plot IoU
plt.subplot(2, 2, 2)
plt.plot(range(1, num_epochs + 1), train_ious, label='Train IoU', color='blue')
plt.plot(range(1, num_epochs + 1), val_ious, label='Validation IoU', color='red')
plt.xlabel('Epochs')
plt.ylabel('IoU')
plt.title('IoU over Epochs')
plt.legend()

# Plot Dice
plt.subplot(2, 2, 3)
plt.plot(range(1, num_epochs + 1), train_dices, label='Train Dice', color='blue')
plt.plot(range(1, num_epochs + 1), val_dices, label='Validation Dice', color='red')
plt.xlabel('Epochs')
plt.ylabel('Dice Coefficient')
plt.title('Dice Coefficient over Epochs')
plt.legend()

# Plot Accuracy
plt.subplot(2, 2, 4)
plt.plot(range(1, num_epochs + 1), train_accuracies, label='Train Accuracy', color='blue')
plt.plot(range(1, num_epochs + 1), val_accuracies, label='Validation Accuracy', color='red')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy over Epochs')
plt.legend()

plt.tight_layout()
plt.show()


