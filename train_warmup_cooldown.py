import os
from pathlib import Path
import shutil
import numpy as np
from PIL import Image
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
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
TILE_SIZE        = 1024
BATCH_SIZE       = 14
LEARNING_RATE    = 1e-4
MIN_LEARNING_RATE= 1e-5
NUM_EPOCHS       = 300
WARMUP_EPOCHS    = 30      # ← Number of epochs to linearly warm up
FACTOR           = 0.7      # ← LR reduction factor
PATIENCE         = 6        # ← LR reduction patience
SEED             = 100
NUM_WORKERS      = 0
IMG_DIR          = Path("data/tifs")
MSK_DIR          = Path("data/handcrafted_masks")
OUT_IMG          = Path("tiles/images")
OUT_MSK          = Path("tiles/masks")
INDEX_CSV        = Path("data/index.csv")

# ─── DEVICE SELECTION ───────────────────────────────────────────────────────
def select_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("✅ Using CUDA (GPU)")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
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

# ─── TILING FUNCTION ────────────────────────────────────────────────────────
def create_tiles(img_dir, msk_dir, out_img, out_msk, tile_size):
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
                name = f"{stem}__{i:05d}_{j:05d}"
                tile_img = img_arr[i:i+tile_size, j:j+tile_size]
                tile_msk = msk_arr[i:i+tile_size, j:j+tile_size]

                # ensure 8-bit RGB
                if tile_img.dtype != np.uint8:
                    tile_img = (tile_img >> 8).astype(np.uint8)
                if tile_img.ndim == 2:
                    tile_img = np.stack([tile_img]*3, axis=-1)

                Image.fromarray(tile_img).save(out_img / f"{name}.png")
                Image.fromarray(tile_msk, mode="L").save(out_msk / f"{name}.png")

    print("✅ Tiling complete!")

# ─── SPLIT TILES BASED ON INDEX.CSV ─────────────────────────────────────────
def split_tiles_by_index(img_tiles_dir, msk_tiles_dir, index_csv):
    df = pd.read_csv(index_csv)
    train_targets = df[df['training or validation']=='training']['target'].unique()
    val_targets   = df[df['training or validation']=='validation']['target'].unique()
    all_imgs      = sorted(img_tiles_dir.glob("*.png"))

    train_imgs, val_imgs = [], []
    for img_path in all_imgs:
        orig = img_path.stem.split('__')[0]
        if orig in val_targets:
            val_imgs.append(img_path)
        elif orig in train_targets:
            train_imgs.append(img_path)
        else:
            print(f"⚠️ No target for {orig}, skipping")

    train_msks = [msk_tiles_dir / f"{p.stem}.png" for p in train_imgs]
    val_msks   = [msk_tiles_dir / f"{p.stem}.png" for p in val_imgs]
    print(f"→ {len(train_imgs)} training / {len(val_imgs)} validation tiles")

    missing = [p for p in train_msks + val_msks if not p.exists()]
    if missing:
        print(f"⚠️ Missing mask files for: {missing}")
    return train_imgs, train_msks, val_imgs, val_msks

# ─── DATASET CLASS ─────────────────────────────────────────────────────────
class TiledSegmentationDataset(Dataset):
    def __init__(self, images, masks, tile_size, transform=None):
        self.images    = images
        self.masks     = masks
        self.tile_size = tile_size
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = str(self.images[idx])
        msk_path = str(self.masks[idx])
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        msk = cv2.imread(msk_path, cv2.IMREAD_GRAYSCALE)
        if img is None or msk is None:
            raise RuntimeError(f"Failed to load {img_path} or {msk_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform:
            aug = self.transform(image=img, mask=msk)
            img, msk = aug['image'], aug['mask']

        msk = msk.unsqueeze(0).float() / 255.0
        return img, msk

# ─── LOSS FUNCTIONS ────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha, self.gamma, self.reduction = alpha, gamma, reduction

    def forward(self, logits, targets):
        bce   = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        pt    = targets*probs + (1-targets)*(1-probs)
        w     = self.alpha * (1-pt).pow(self.gamma)
        loss  = w * bce
        return loss.mean() if self.reduction=="mean" else loss.sum()

class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce     = nn.BCEWithLogitsLoss()
        self.dice    = smp.losses.DiceLoss(mode="binary")
        self.focal   = FocalLoss(alpha=0.25, gamma=2.0)
        self.tversky = smp.losses.TverskyLoss(mode="binary", alpha=0.7)

    def forward(self, outputs, targets):
        return (
            self.bce(outputs, targets)
          + self.dice(outputs, targets)
          + 0.5 * self.focal(outputs, targets)
          + 0.5 * self.tversky(outputs, targets)
        )

# ─── TRAIN & VALIDATE ──────────────────────────────────────────────────────
def train_one_epoch(model, loader, optim, loss_fn, device):
    model.train()
    running, total_iou, total_dice, total_acc, batches = 0, 0, 0, 0, 0
    for imgs, msks in tqdm(loader, desc="Training"):
        imgs, msks = imgs.to(device), msks.to(device)
        optim.zero_grad()
        outs = model(imgs)
        loss = loss_fn(outs, msks)
        loss.backward()
        optim.step()
        running += loss.item()

        with torch.no_grad():
            preds = (torch.sigmoid(outs)>0.5).float()
            inter = (preds*msks).sum((1,2,3))
            union= ((preds+msks)>0).sum((1,2,3))
            total_iou  += (inter/(union+1e-7)).mean().item()
            total_dice += (2*inter/(preds.sum((1,2,3))+msks.sum((1,2,3))+1e-7)).mean().item()
            total_acc  += (preds==msks).float().mean().item()
            batches    += 1

    return {
        "loss": running/batches,
        "iou":  total_iou/batches,
        "dice": total_dice/batches,
        "accuracy": total_acc/batches
    }

def validate(model, loader, loss_fn, device):
    model.eval()
    val_loss, total_iou, total_dice, total_acc, batches = 0, 0, 0, 0, 0
    with torch.no_grad():
        for imgs, msks in tqdm(loader, desc="Validation"):
            imgs, msks = imgs.to(device), msks.to(device)
            outs = model(imgs)
            val_loss += loss_fn(outs, msks).item()

            preds = (torch.sigmoid(outs)>0.5).float()
            inter = (preds*msks).sum((1,2,3))
            union= ((preds+msks)>0).sum((1,2,3))
            total_iou  += (inter/(union+1e-7)).mean().item()
            total_dice += (2*inter/(preds.sum((1,2,3))+msks.sum((1,2,3))+1e-7)).mean().item()
            total_acc  += (preds==msks).float().mean().item()
            batches    += 1

    return {
        "loss":     val_loss/batches,
        "iou":      total_iou/batches,
        "dice":     total_dice/batches,
        "accuracy": total_acc/batches
    }

# ─── WARMUP + PLATEAU SCHEDULERS ───────────────────────────────────────────
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau

def get_schedulers(optimizer):
    # Linear warmup: epoch 0→WARMUP_EPOCHS linearly ramps 0→1
    def lr_lambda(epoch):
        if epoch < WARMUP_EPOCHS:
            return float(epoch+1) / float(WARMUP_EPOCHS)
        return 1.0

    warmup = LambdaLR(optimizer, lr_lambda=lr_lambda)
    plateau = ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=FACTOR,
        patience=PATIENCE,
        verbose=True,
        min_lr=MIN_LEARNING_RATE
    )
    return warmup, plateau

# ─── MAIN EXECUTION ────────────────────────────────────────────────────────
def main():
    device = select_device()
    set_seed(SEED)

    create_tiles(IMG_DIR, MSK_DIR, OUT_IMG, OUT_MSK, TILE_SIZE)
    train_imgs, train_msks, val_imgs, val_msks = split_tiles_by_index(OUT_IMG, OUT_MSK, INDEX_CSV)

    train_tf = A.Compose([
        A.HorizontalFlip(p=0.4), A.RandomRotate90(p=0.4),
        A.RandomBrightnessContrast(p=0.4),
        A.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
        ToTensorV2()],
        additional_targets={'mask':'mask'}
    )
    val_tf = A.Compose([
        A.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
        ToTensorV2()],
        additional_targets={'mask':'mask'}
    )

    train_ds = TiledSegmentationDataset(train_imgs, train_msks, TILE_SIZE, transform=train_tf)
    val_ds   = TiledSegmentationDataset(val_imgs,   val_msks,   TILE_SIZE, transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)

    model = smp.Unet("resnet34", encoder_weights="imagenet", in_channels=3, classes=1).to(device)
    loss_fn   = CombinedLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    warmup_sched, plateau_sched = get_schedulers(optimizer)

    best_iou = 0.0
    history = {k:[] for k in ["train_loss","val_loss","train_iou","val_iou","train_dice","val_dice","train_acc","val_acc"]}

    for epoch in range(NUM_EPOCHS):
        print(f"\n=== Epoch {epoch+1}/{NUM_EPOCHS} ===")
        tm = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        vm = validate(model, val_loader,   loss_fn, device)
        history["train_loss"].append(tm["loss"])
        history["val_loss"].append(vm["loss"])
        history["train_iou"].append(tm["iou"])
        history["val_iou"].append(vm["iou"])
        history["train_dice"].append(tm["dice"])
        history["val_dice"].append(vm["dice"])
        history["train_acc"].append(tm["accuracy"])
        history["val_acc"].append(vm["accuracy"])

        # step schedulers
        if epoch < WARMUP_EPOCHS:
            warmup_sched.step()
        else:
            plateau_sched.step(vm["iou"])

        lr = optimizer.param_groups[0]["lr"]
        print(f"Train IoU: {tm['iou']:.4f} | Val IoU: {vm['iou']:.4f} | LR: {lr:.6f}")

        if vm["iou"] > best_iou:
            best_iou = vm["iou"]
            torch.save(model.state_dict(), "best_model.pth")
            print(f"✅ Saved best model (IoU={best_iou:.4f})")

    # Plotting (unchanged)
    plt.figure(figsize=(12,10))
    plt.subplot(2,2,1)
    plt.plot(range(1,NUM_EPOCHS+1), history["train_loss"], label="Train Loss")
    plt.plot(range(1,NUM_EPOCHS+1), history["val_loss"],   label="Val Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()

    plt.subplot(2,2,2)
    plt.plot(range(1,NUM_EPOCHS+1), history["train_iou"], label="Train IoU")
    plt.plot(range(1,NUM_EPOCHS+1), history["val_iou"],   label="Val IoU")
    plt.xlabel("Epoch"); plt.ylabel("IoU"); plt.legend()

    plt.subplot(2,2,3)
    plt.plot(range(1,NUM_EPOCHS+1), history["train_dice"], label="Train Dice")
    plt.plot(range(1,NUM_EPOCHS+1), history["val_dice"],   label="Val Dice")
    plt.xlabel("Epoch"); plt.ylabel("Dice"); plt.legend()

    plt.subplot(2,2,4)
    plt.plot(range(1,NUM_EPOCHS+1), history["train_acc"], label="Train Acc")
    plt.plot(range(1,NUM_EPOCHS+1), history["val_acc"],   label="Val Acc")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
