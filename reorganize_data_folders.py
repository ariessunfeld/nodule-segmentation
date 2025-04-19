import shutil
from pathlib import Path

def reorganize_data(data_dir: Path):
    # 1) Create target folders
    tifs_dir  = data_dir / 'tifs'
    masks_dir = data_dir / 'handcrafted_masks'
    tifs_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    # 2) Iterate each subfolder (skip the new ones)
    for sub in data_dir.iterdir():
        if not sub.is_dir() or sub.name in {'tifs', 'handcrafted_masks'}:
            continue

        # 3) Find exactly one .tif and one .png
        tif_list = list(sub.glob('*.tif'))
        png_list = list(sub.glob('*.png'))
        if len(tif_list) != 1 or len(png_list) != 1:
            print(f"⚠️  Skipping '{sub.name}': found {len(tif_list)} .tif, {len(png_list)} .png")
            continue

        tif_path = tif_list[0]
        png_path = png_list[0]

        # 4) Copy & rename based on the parent-folder name
        new_tif = tifs_dir  / f"{sub.name}.tif"
        new_png = masks_dir / f"{sub.name}.png"

        shutil.copy(str(tif_path), str(new_tif))
        shutil.copy(str(png_path), str(new_png))
        print(f"✔️  Copy '{sub.name}' → {new_tif.name}, {new_png.name}")

    print("\nReorganization complete.")

if __name__ == "__main__":
    base = Path(__file__).parent / 'data'
    reorganize_data(base)