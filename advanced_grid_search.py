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
import itertools
import time
import json


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


class WeightedCombinedLoss(nn.Module):
    """Weighted combination of multiple losses."""
    def __init__(self, dice_weight=1.0, bce_weight=1.0, focal_weight=0.0):
        super().__init__()
        self.dice = smp.losses.DiceLoss(mode='binary')
        self.bce = nn.BCEWithLogitsLoss()
        self.focal = smp.losses.FocalLoss(mode='binary')
        
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.focal_weight = focal_weight
    
    def forward(self, outputs, targets):
        loss = 0
        if self.dice_weight > 0:
            loss += self.dice_weight * self.dice(outputs, targets)
        if self.bce_weight > 0:
            loss += self.bce_weight * self.bce(outputs, targets)
        if self.focal_weight > 0:
            loss += self.focal_weight * self.focal(outputs, targets)
        return loss


class TverskyLoss(nn.Module):
    """Tversky loss - allows control over false positives/negatives."""
    def __init__(self, alpha=0.5, beta=0.5, smooth=1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
    
    def forward(self, outputs, targets):
        outputs = torch.sigmoid(outputs)
        
        # True Positives, False Positives & False Negatives
        tp = (outputs * targets).sum()
        fp = ((1 - targets) * outputs).sum()
        fn = (targets * (1 - outputs)).sum()
        
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + 
                                       self.beta * fn + self.smooth)
        return 1 - tversky


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


def get_optimizer(model, optimizer_name, learning_rate, weight_decay=1e-4):
    """Get optimizer by name with weight decay."""
    if optimizer_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=learning_rate, 
                                weight_decay=weight_decay)
    elif optimizer_name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=learning_rate, 
                              momentum=0.9, weight_decay=weight_decay)
    elif optimizer_name == "radam":
        try:
            from torch.optim import RAdam
            return RAdam(model.parameters(), lr=learning_rate, 
                         weight_decay=weight_decay)
        except ImportError:
            print("RAdam not available, falling back to Adam")
            return torch.optim.Adam(model.parameters(), lr=learning_rate)
    else:
        return torch.optim.Adam(model.parameters(), lr=learning_rate)


def get_scheduler(optimizer, scheduler_name, steps_per_epoch, max_epochs):
    """Get learning rate scheduler."""
    if scheduler_name == "reduce_on_plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5, 
            verbose=True, min_lr=1e-6)
    elif scheduler_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max_epochs, eta_min=1e-6)
    elif scheduler_name == "one_cycle":
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=optimizer.param_groups[0]['lr'] * 10,
            steps_per_epoch=steps_per_epoch, epochs=max_epochs,
            pct_start=0.2, div_factor=10, final_div_factor=1000)
    else:
        return None


def train_one_epoch(model, dataloader, optimizer, loss_fn, device, scheduler=None):
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
        
        if scheduler and scheduler.__class__.__name__ == "OneCycleLR":
            scheduler.step()

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


def train_with_advanced_params(
    learning_rate, tile_size, batch_size,
    optimizer_name="adamw", weight_decay=1e-4,
    dice_weight=1.0, bce_weight=1.0, focal_weight=0.0,
    scheduler_name="reduce_on_plateau", 
    encoder_name="efficientnet-b4",
    num_epochs=20, early_stopping_patience=10,
    seed=100):
    """Train model with advanced hyperparameters."""
    
    # Set seed for reproducibility
    set_seed(seed)
    
    # Device setup
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
        print("Using MPS")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
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
        A.ElasticTransform(p=0.3, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
        A.GridDistortion(p=0.3),
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
    try:
        model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights="imagenet",
            in_channels=3,
            classes=1
        ).to(device)
    except Exception as e:
        print(f"Error with encoder {encoder_name}: {e}")
        print("Falling back to resnet34")
        model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=3,
            classes=1
        ).to(device)
    
    # Loss function with coefficients
    loss_fn = WeightedCombinedLoss(dice_weight, bce_weight, focal_weight)
    
    # Optimizer
    optimizer = get_optimizer(model, optimizer_name, learning_rate, weight_decay)
    
    # Scheduler
    scheduler = get_scheduler(optimizer, scheduler_name, 
                            len(train_loader), num_epochs)
    
    # Early stopping
    best_iou = 0.0
    epochs_without_improvement = 0
    train_losses, val_losses = [], []
    train_ious, val_ious = [], []
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        
        train_metrics = train_one_epoch(model, train_loader, optimizer, loss_fn, device, scheduler)
        val_metrics = validate(model, val_loader, loss_fn, device)
        
        train_losses.append(train_metrics['loss'])
        val_losses.append(val_metrics['loss'])
        train_ious.append(train_metrics['iou'])
        val_ious.append(val_metrics['iou'])
        
        # Update scheduler
        if scheduler_name == "reduce_on_plateau" and scheduler:
            scheduler.step(val_metrics['iou'])
        elif scheduler and scheduler_name != "one_cycle":
            scheduler.step()
        
        print(f"Train Loss: {train_metrics['loss']:.4f} | Train IoU: {train_metrics['iou']:.4f}")
        print(f"Val Loss: {val_metrics['loss']:.4f} | Val IoU: {val_metrics['iou']:.4f}")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Early stopping check
        if val_metrics['iou'] > best_iou:
            best_iou = val_metrics['iou']
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            
        if epochs_without_improvement >= early_stopping_patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break
    
    return {
        'best_val_iou': best_iou,
        'learning_rate': learning_rate,
        'tile_size': tile_size,
        'batch_size': batch_size,
        'optimizer': optimizer_name,
        'weight_decay': weight_decay,
        'dice_weight': dice_weight,
        'bce_weight': bce_weight,
        'focal_weight': focal_weight,
        'scheduler': scheduler_name,
        'encoder': encoder_name,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_ious': train_ious,
        'val_ious': val_ious
    }


def advanced_grid_search():
    """Perform advanced grid search."""
    # Extended hyperparameter ranges
    learning_rates = [1e-4, 2e-4, 5e-4]
    tile_sizes = [256, 384]
    batch_sizes = [8, 16, 24]
    optimizers = ["adamw", "sgd", "radam"]
    weight_decays = [1e-4, 1e-3]
    loss_coefficients = [
        (1.0, 1.0, 0.0),  # dice + bce
        (1.0, 0.5, 0.5),  # dice + 0.5*bce + 0.5*focal
        (0.7, 0.3, 0.5),  # custom weights
    ]
    schedulers = ["reduce_on_plateau", "cosine", "one_cycle"]
    encoders = ["resnet34", "efficientnet-b4", "resnet50"]
    
    results = []
    best_result = None
    best_iou = 0.0
    
    # Generate all combinations
    all_combinations = list(itertools.product(
        learning_rates, tile_sizes, batch_sizes, optimizers,
        weight_decays, loss_coefficients, schedulers, encoders
    ))
    
    # Randomly sample configurations to try
    num_trials = min(50, len(all_combinations))
    trial_combinations = random.sample(all_combinations, num_trials)
    
    start_time = time.time()
    
    for i, combo in enumerate(trial_combinations):
        lr, tile_size, batch_size, optimizer, weight_decay, (dice_w, bce_w, focal_w), scheduler, encoder = combo
        
        print(f"\n{'='*50}")
        print(f"Trial {i+1}/{num_trials}")
        print(f"Configuration: LR={lr}, Tile={tile_size}, Batch={batch_size}, "
              f"Opt={optimizer}, WD={weight_decay}, "
              f"Dice={dice_w}, BCE={bce_w}, Focal={focal_w}, "
              f"Scheduler={scheduler}, Encoder={encoder}")
        print(f"{'='*50}")
        
        result = train_with_advanced_params(
            lr, tile_size, batch_size,
            optimizer_name=optimizer, weight_decay=weight_decay,
            dice_weight=dice_w, bce_weight=bce_w, focal_weight=focal_w,
            scheduler_name=scheduler, encoder_name=encoder,
            num_epochs=30  # Set to 30 for grid search
        )
        results.append(result)
        
        if result['best_val_iou'] > best_iou:
            best_iou = result['best_val_iou']
            best_result = result
        
        print(f"Result: Best Val IoU={result['best_val_iou']:.4f}")
        elapsed_time = time.time() - start_time
        print(f"Time elapsed: {elapsed_time/60:.2f} minutes")
    
    return results, best_result


if __name__ == "__main__":
    print("Starting advanced grid search...")
    all_results, best_result = advanced_grid_search()
    
    print("\n=== BEST RESULT ===")
    print(f"Best Val IoU: {best_result['best_val_iou']:.4f}")
    print(f"Learning Rate: {best_result['learning_rate']}")
    print(f"Tile Size: {best_result['tile_size']}")
    print(f"Batch Size: {best_result['batch_size']}")
    print(f"Optimizer: {best_result['optimizer']}")
    print(f"Weight Decay: {best_result['weight_decay']}")
    print(f"Dice Weight: {best_result['dice_weight']}")
    print(f"BCE Weight: {best_result['bce_weight']}")
    print(f"Focal Weight: {best_result['focal_weight']}")
    print(f"Scheduler: {best_result['scheduler']}")
    print(f"Encoder: {best_result['encoder']}")
    
    # Save results to file
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    
    # Save all results
    with open(f'grid_search_results_{timestamp}.json', 'w') as f:
        json_results = [{
            'best_val_iou': r['best_val_iou'],
            'learning_rate': r['learning_rate'],
            'tile_size': r['tile_size'],
            'batch_size': r['batch_size'],
            'optimizer': r['optimizer'],
            'weight_decay': r['weight_decay'],
            'dice_weight': r['dice_weight'],
            'bce_weight': r['bce_weight'],
            'focal_weight': r['focal_weight'],
            'scheduler': r['scheduler'],
            'encoder': r['encoder']
        } for r in all_results]
        json.dump(json_results, f, indent=4)
    
    # Save best result
    with open(f'best_result_{timestamp}.json', 'w') as f:
        best_json = {
            'best_val_iou': best_result['best_val_iou'],
            'learning_rate': best_result['learning_rate'],
            'tile_size': best_result['tile_size'],
            'batch_size': best_result['batch_size'],
            'optimizer': best_result['optimizer'],
            'weight_decay': best_result['weight_decay'],
            'dice_weight': best_result['dice_weight'],
            'bce_weight': best_result['bce_weight'],
            'focal_weight': best_result['focal_weight'],
            'scheduler': best_result['scheduler'],
            'encoder': best_result['encoder']
        }
        json.dump(best_json, f, indent=4)
    
    print(f"\nResults saved to 'grid_search_results_{timestamp}.json' and 'best_result_{timestamp}.json'")