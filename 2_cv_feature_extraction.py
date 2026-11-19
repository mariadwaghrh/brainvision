import os
import glob
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colormaps

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # project folder, works anywhere
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

SIZE, RADIUS = 128, 54

# Circular ROI mask (segmentation: keep the head, drop the background)
MASK = np.zeros((SIZE, SIZE), dtype=np.uint8)
cv2.circle(MASK, (SIZE // 2, SIZE // 2), RADIUS, 255, -1)

# The 'jet' colormap as an RGB lookup table, used to turn colours back into values
JET = (colormaps["jet"](np.linspace(0, 1, 256))[:, :3] * 255).astype(np.float32)

# HOG is kept for the demo figure only: it mostly describes the head outline,
# and adding it to the features lowered cross-validated accuracy.
hog = cv2.HOGDescriptor(_winSize=(128, 128), _blockSize=(32, 32),
                        _blockStride=(16, 16), _cellSize=(16, 16), _nbins=9)


def activity_map(img_bgr):
    """Invert the jet colormap: each pixel colour -> brain activity value in [0, 1].
    Pixels that are not on the colormap (outline, background) become NaN."""
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).reshape(-1, 3).astype(np.float32)
    dist = ((rgb[:, None, :] - JET[None]) ** 2).sum(-1)
    values = dist.argmin(1) / 255.0
    values[dist.min(1) > 1600] = np.nan
    values = values.reshape(SIZE, SIZE)
    values[MASK == 0] = np.nan
    return values


def extract_vision_features(img_path, save_demo=False):
    img = cv2.resize(cv2.imread(img_path), (SIZE, SIZE))
    masked_img = cv2.bitwise_and(img, img, mask=MASK)

    # 1. HSV colour histograms for each hemisphere + their difference (192 features)
    hsv = cv2.cvtColor(masked_img, cv2.COLOR_BGR2HSV)
    half = SIZE // 2
    hist_left = cv2.calcHist([hsv[:, :half]], [0, 1], MASK[:, :half], [8, 8], [0, 180, 0, 256])
    hist_right = cv2.calcHist([hsv[:, half:]], [0, 1], MASK[:, half:], [8, 8], [0, 180, 0, 256])
    cv2.normalize(hist_left, hist_left)
    cv2.normalize(hist_right, hist_right)
    color_features = np.hstack([hist_left.flatten(), hist_right.flatten(), (hist_left - hist_right).flatten()])

    # 2. Activity map: colormap inversion, then a 16x16 grid (256 features)
    act = activity_map(img)
    act_filled = np.where(np.isnan(act), 0.5, act)  # 0.5 = average activity (z = 0)
    grid = cv2.resize(act_filled, (16, 16), interpolation=cv2.INTER_AREA)

    # 3. Mirror asymmetry: left half minus the mirrored right half (128 features)
    mirror = act_filled - act_filled[:, ::-1]
    mirror_grid = cv2.resize(mirror[:, :half], (8, 16), interpolation=cv2.INTER_AREA)

    if save_demo:
        gray = cv2.cvtColor(masked_img, cv2.COLOR_BGR2GRAY)
        fig, axes = plt.subplots(1, 5, figsize=(16, 3.6))
        axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); axes[0].set_title("1. Raw brain topomap")
        axes[1].imshow(cv2.cvtColor(masked_img, cv2.COLOR_BGR2RGB)); axes[1].set_title("2. Circular ROI mask")
        axes[2].imshow(hsv[:, :, 0], cmap="hsv"); axes[2].axvline(half, color="w", ls="--", lw=2); axes[2].set_title("3. HSV hue, left | right")
        axes[3].imshow(grid, cmap="jet", vmin=0, vmax=1); axes[3].set_title("4. Activity map (16x16 grid)")
        lim = np.abs(mirror).max() or 1
        axes[4].imshow(mirror, cmap="RdBu_r", vmin=-lim, vmax=lim); axes[4].set_title("5. Mirror asymmetry")
        for a in axes: a.axis("off")
        plt.suptitle("Computer Vision Pipeline: Brain Topomap Analysis", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(BASE_DIR, "1_cv_pipeline_demo.png"), dpi=150)
        plt.close(fig)
        _ = hog.compute(gray)  # shown in the website demo, not used as a feature

    return np.hstack([color_features, grid.ravel(), mirror_grid.ravel()])


if __name__ == "__main__":
    print("[2/3] Running OpenCV Feature Extraction...")
    X, y, file_paths = [], [], []
    for label_idx, cls_name in enumerate(["left_hand", "right_hand"]):
        for img_path in sorted(glob.glob(os.path.join(DATASET_DIR, cls_name, "*.png"))):
            X.append(extract_vision_features(img_path, save_demo=(len(X) == 0)))
            y.append(label_idx)
            file_paths.append(img_path)

    X, y = np.array(X), np.array(y)
    output_file = os.path.join(BASE_DIR, "cv_features.npz")
    np.savez(output_file, X=X, y=y, paths=file_paths)
    print(f"Extracted {X.shape[1]} CV features (192 colour + 256 activity + 128 asymmetry) "
          f"from {X.shape[0]} images -> Saved to {output_file}")
