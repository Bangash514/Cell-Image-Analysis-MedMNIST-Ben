# BEN 2026-11-01
# Change aim: Explore class balance and demonstrate CLAHE, Gaussian and Otsu preprocessing.
# File: 01_explore_preprocess.py
"""Step 1: explore the data, check class balance, show preprocessing effects."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage import exposure, filters, color
from common import load_split, CLASS_NAMES

os.makedirs("results", exist_ok=True)
X_tr, y_tr = load_split("train")
X_va, y_va = load_split("val")
X_te, y_te = load_split("test")
print("train/val/test sizes:", len(X_tr), len(X_va), len(X_te))

# class balance
counts = np.bincount(y_tr, minlength=len(CLASS_NAMES))
plt.figure(figsize=(7, 3.5))
plt.bar(range(len(counts)), counts)
plt.xticks(range(len(counts)), CLASS_NAMES, rotation=40, ha="right")
plt.ylabel("training images")
plt.title("BloodMNIST class balance (train)")
plt.tight_layout()
plt.savefig("results/class_balance.png", dpi=150)
plt.close()

# example images per class
fig, axes = plt.subplots(2, 8, figsize=(14, 4))
for c in range(len(CLASS_NAMES)):
    idx = np.where(y_tr == c)[0][:2]
    for r, i in enumerate(idx):
        axes[r, c].imshow(X_tr[i])
        axes[r, c].axis("off")
        if r == 0:
            axes[r, c].set_title(CLASS_NAMES[c], fontsize=7)
plt.tight_layout()
plt.savefig("results/examples.png", dpi=150)
plt.close()

# preprocessing demo: contrast enhancement + smoothing + Otsu threshold
img = X_tr[0]
gray = color.rgb2gray(img)
eq = exposure.equalize_adapthist(gray, clip_limit=0.02)
sm = filters.gaussian(eq, sigma=1)
th = filters.threshold_otsu(sm)
fig, axes = plt.subplots(1, 4, figsize=(11, 3))
for ax, im, t in zip(axes, [img, gray, sm, sm > th],
                     ["original", "grayscale", "CLAHE + Gaussian", f"Otsu mask (t={th:.2f})"]):
    ax.imshow(im, cmap=None if im.ndim == 3 else "gray")
    ax.set_title(t, fontsize=8)
    ax.axis("off")
plt.tight_layout()
plt.savefig("results/preprocessing_demo.png", dpi=150)
plt.close()

with open("results/class_counts.txt", "w") as f:
    for n, c in zip(CLASS_NAMES, counts):
        f.write(f"{n}: {c}\n")
print("saved figures to results/")
