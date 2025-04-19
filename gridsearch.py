import numpy as np
import torch
import cv2
import shutil
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from tqdm import tqdm
import random
import torch.nn as nn


def set_seed(seed: int):
    """Set seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.manual_seed(seed)


class CombinedLoss(nn.Module):
    """Combined Dice and BCE loss."""
    def __init__(self):
        super().__init__()
        self.dice = smp.losses.DiceLoss(mode='binary')
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, outputs, targets):
        return self.dice(outputs, targets) + self.bce(outputs, targets)


class TiledSegmentationDataset(Dataset):
    """Dataset for tiled segmentation."""
    def __init__(self, images, masks, tile_size, transform=None):
        self.images = images
        self.masks = masks
        self.tile_size = tile_size
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = cv2.imread(self.images[idx], cv2.IMREAD_COLOR)
        mask = cv2.imread(self.masks[idx], cv2.IMREAD_GRAYSCALE)

        if img is None or mask is None:
            raise RuntimeError(f"Failed to load {self.images[idx]} or its mask")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img, mask = augmented['image'], augmented['mask']

        mask = mask.unsqueeze(0).float() / 255.0
        return img, mask


def generate_tiles(tile_size: int):
    """Generate tiles with the specified tile size."""
    # Paths
    IMG_DIR = Path("data/tifs")
    MSK_DIR = Path("data/handcrafted_masks")
    OUT_IMG = Path("tiles/images")
    OUT_MSK = Path("tiles/masks")

    # Clean existing tiles
    if OUT_IMG.exists():
        shutil.rmtree(OUT_IMG)
    if OUT_MSK.exists():
        shutil.rmtree(OUT_MSK)

    OUT_IMG.mkdir(parents=True, exist_ok=True)
    OUT_MSK.mkdir(parents=True, exist_ok=True)

    # Helper function for mask verification
    def verify_mask(arr: np.ndarray) -> np.ndarray:
        if arr.ndim != 2 or not np.array_equal(np.unique(arr), [0, 255]):
            gray = Image.fromarray(arr).convert("L")
            a = np.array(gray)
            return (a >= 255).astype(np.uint8) * 255
        return arr

    # Tile generation
    for img_path in sorted(IMG_DIR.glob("*.tif")):
        stem = img_path.stem
        mask_path = MSK_DIR / f"{stem}.png"
        if not mask_path.exists():
            print(f"⚠️ no mask for {stem}, skipping")
            continue

        img_arr = np.array(Image.open(img_path))
        msk_arr = verify_mask(np.array(Image.open(mask_path)))

        h, w = msk_arr.shape[:2]
        for i in range(0, h, tile_size):
            for j in range(0, w, tile_size):
                name = f"{stem}_{i//tile_size:02d}_{j//tile_size:02d}"

                tile_img = img_arr[i:i+tile_size, j:j+tile_size]
                tile_msk = msk_arr[i:i+tile_size, j:j+tile_size]

                if tile_img.dtype != np.uint8:
                    tile_img8 = (tile_img >> 8).astype(np.uint8)
                else:
                    tile_img8 = tile_img

                if tile_img8.ndim == 2:
                    tile_img8 = np.stack([tile_img8]*3, axis=-1)

                Image.fromarray(tile_img8).save(OUT_IMG / f"{name}.png")
                Image.fromarray(tile_msk, mode="L").save(OUT_MSK / f"{name}.png")

    return OUT_IMG, OUT_MSK


def train_one_epoch(model, dataloader, optimizer, loss_fn, device):
    """Train model for one epoch."""
    model.train()
    running_loss = 0.0
    total_iou = 0.0
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

            total_iou += iou
            num_batches += 1

    return {
        "loss": running_loss / num_batches,
        "iou": total_iou / num_batches
    }


def validate(model, dataloader, loss_fn, device):
    """Validate model."""
    model.eval()
    val_loss = 0.0
    total_iou = 0.0
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

            total_iou += iou
            num_batches += 1

    return {
        "loss": val_loss / num_batches,
        "iou": total_iou / num_batches
    }


def train_with_params(learning_rate, tile_size, batch_size, num_epochs=30, seed=100):
    """Train model with specified hyperparameters and return best IoU."""
    # Set seed for reproducibility
    set_seed(seed)
    
    # Device setup
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    # Generate tiles with specified size
    IMG_TILES, MSK_TILES = generate_tiles(tile_size)
    
    # Split into train and validation
    all_imgs = sorted(IMG_TILES.glob("*.png"))
    val_images = [p for p in all_imgs if "achnacarry" in p.stem.lower() or 
                  "ayre_of_tonga" in p.stem.lower() or "addiewell" in p.stem.lower()]
    train_images = [p for p in all_imgs if "achnacarry" not in p.stem.lower() and 
                    "ayre_of_tonga" not in p.stem.lower() and "addiewell" not in p.stem.lower()]
    
    val_masks = [MSK_TILES / f"{p.stem}.png" for p in val_images]
    train_masks = [MSK_TILES / f"{p.stem}.png" for p in train_images]
    
    # Create transforms
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
    
    # Create datasets and dataloaders
    train_dataset = TiledSegmentationDataset(train_images, train_masks, tile_size, train_transform)
    val_dataset = TiledSegmentationDataset(val_images, val_masks, tile_size, val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                            num_workers=0, pin_memory=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                          num_workers=0, pin_memory=True)
    
    # Initialize model
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1
    ).to(device)
    
    # Initialize loss and optimizer
    loss_fn = CombinedLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    best_iou = 0.0
    train_losses, val_losses = [], []
    train_ious, val_ious = [], []
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        
        train_metrics = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_metrics = validate(model, val_loader, loss_fn, device)
        
        train_losses.append(train_metrics['loss'])
        val_losses.append(val_metrics['loss'])
        train_ious.append(train_metrics['iou'])
        val_ious.append(val_metrics['iou'])
        
        print(f"Train Loss: {train_metrics['loss']:.4f} | Train IoU: {train_metrics['iou']:.4f}")
        print(f"Val Loss: {val_metrics['loss']:.4f} | Val IoU: {val_metrics['iou']:.4f}")
        
        if val_metrics['iou'] > best_iou:
            best_iou = val_metrics['iou']
    
    # Return best IoU and all hyperparameters
    return {
        'best_val_iou': best_iou,
        'learning_rate': learning_rate,
        'tile_size': tile_size,
        'batch_size': batch_size,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_ious': train_ious,
        'val_ious': val_ious
    }


# Example grid search implementation
def grid_search():
    """Perform grid search over hyperparameters."""
    # Define hyperparameter ranges
    learning_rates = [4e-5, 1e-4, 2e-4]
    tile_sizes = [128, 256, 512]
    batch_sizes = [4, 8, 16]
    
    results = []
    best_result = None
    best_iou = 0.0
    
    # Perform grid search
    for lr in learning_rates:
        for tile_size in tile_sizes:
            for batch_size in batch_sizes:
                print(f"\nTesting: LR={lr}, Tile Size={tile_size}, Batch Size={batch_size}")
                
                result = train_with_params(lr, tile_size, batch_size)
                results.append(result)
                
                if result['best_val_iou'] > best_iou:
                    best_iou = result['best_val_iou']
                    best_result = result
                
                print(f"Result: Best Val IoU={result['best_val_iou']:.4f}")
    
    return results, best_result


if __name__ == "__main__":
    # Run grid search
    all_results, best_result = grid_search()
    
    print("\n=== BEST RESULT ===")
    print(f"Best Val IoU: {best_result['best_val_iou']:.4f}")
    print(f"Learning Rate: {best_result['learning_rate']}")
    print(f"Tile Size: {best_result['tile_size']}")
    print(f"Batch Size: {best_result['batch_size']}")