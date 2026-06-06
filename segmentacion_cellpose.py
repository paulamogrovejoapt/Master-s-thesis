# -*- coding: utf-8 -*-
"""
segmentacion_cellpose.ipynb
===========================
Confocal microscopy image segmentation pipeline for plant root cross-sections.
Uses Cellpose (deep learning) to detect individual cells and extract quantitative
metrics per cell: area, mean intensity, wall intensity, and wall percentage.

Developed for the analysis of Brassicaceae halophyte root cell wall composition
under salt stress conditions (AgroParisTech / EPFL Plant Adaptation Laboratory).

Original Colab file:
https://colab.research.google.com/drive/17ZaTl0zYdOEGAPH5dZOkshNMwzi1zXAa

Instructions:
    1. Run cells in order, top to bottom.
    2. Click the ▶ button on the left of each cell to execute it.
    3. Wait for ✅ before moving to the next cell.
"""

# =============================================================================
# STEP 1 — Install dependencies
# -----------------------------------------------------------------------------
# Run this every time you open a new Colab session.
# Takes ~1 minute.
# =============================================================================

# Force a NumPy version compatible with Numba and Cellpose
!pip install "numpy>=1.24,<2.1" --force-reinstall -q

# Install Cellpose and supporting tools without overwriting NumPy
!pip install "cellpose>=3.0" scipy tifffile pandas matplotlib opencv-python-headless --no-deps -q

print('✅ Installation complete')

# =============================================================================
# STEP 2 — Check GPU availability
# -----------------------------------------------------------------------------
# Colab provides a free GPU. If this prints False, go to:
# Runtime > Change runtime type > T4 GPU > Save
# =============================================================================

import torch
print('GPU available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU model:', torch.cuda.get_device_name(0))
else:
    print('⚠️  No GPU detected — change runtime type to T4 GPU')

# =============================================================================
# STEP 3 — Mount Google Drive and locate the image
# -----------------------------------------------------------------------------
# Upload your TIFF to Google Drive before running this step.
# The cell will print the contents of your Drive root so you can find the file.
# =============================================================================

from google.colab import drive
import os

drive.mount('/content/drive')
print('✅ Google Drive connected')

print('\nContents of MyDrive:')
for f in os.listdir('/content/drive/MyDrive'):
    print(f)

# =============================================================================
# STEP 4 — Load the image
# -----------------------------------------------------------------------------
# Set the filename below to match your TIFF exactly.
# Use the raw, unprocessed grayscale image from the microscope export.
# If your TIFF is inside a subfolder, adjust the path accordingly:
#   e.g. ruta_imagen = '/content/drive/MyDrive/Images/at_NT_1r_meristem.tif'
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
import cv2
import tifffile as tfi

filename = 'at_NT_1r_meristem.tif'  # <-- change to your filename
image_path = f'/content/drive/MyDrive/{filename}'

img_raw = np.squeeze(tfi.imread(image_path))
print(f'Original dimensions: {img_raw.shape}')
print(f'Data type: {img_raw.dtype}')

# Handle multichannel images (e.g. pseudocolored blue exports from Fiji)
if img_raw.ndim == 3:
    n_channels = img_raw.shape[0] if img_raw.shape[0] < img_raw.shape[-1] else img_raw.shape[2]
    print(f'Multichannel image detected — {n_channels} channels')
    img_raw = img_raw[0] if img_raw.shape[0] < img_raw.shape[-1] else img_raw[:, :, 0]
    print(f'Channel selected — new dimensions: {img_raw.shape}')

plt.figure(figsize=(8, 6))
plt.imshow(img_raw, cmap='gray')
plt.title('Loaded image')
plt.axis('off')
plt.show()

# =============================================================================
# STEP 5 — Cell segmentation with Cellpose
# -----------------------------------------------------------------------------
# Tunable parameters:
#   diameter   : estimated cell diameter in pixels. Start with 30.
#                Increase if too many small fragments are detected.
#                Decrease if large cells are missed.
#   model_type : 'cyto3' works best for plant cells. Try 'cyto2' if needed.
#   flow_threshold    : 0.4 (default). Higher = stricter, fewer cells.
#   cellprob_threshold: 0.0 (default). Higher = only high-confidence cells.
# =============================================================================

from cellpose import models

# --- Adjust these parameters ---
diameter   = 30       # estimated cell diameter in pixels
model_type = 'cyto3'  # options: 'cyto3', 'cyto2'
# --------------------------------

# Normalize image to 8-bit for Cellpose
img_8bit = cv2.normalize(img_raw, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

print(f"Loading model '{model_type}'...")
model = models.CellposeModel(gpu=torch.cuda.is_available(), model_type=model_type)

print('Running segmentation...')
masks, flows, styles = model.eval(
    img_8bit,
    diameter=diameter,
    channels=[0, 0],          # grayscale input
    flow_threshold=0.4,
    cellprob_threshold=0.0
)

n_cells = masks.max()
print(f'\n✅ Segmentation complete — {n_cells} cells detected')

# =============================================================================
# STEP 6 — Visualise segmentation result
# =============================================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

axes[0].imshow(img_8bit, cmap='gray')
axes[0].set_title('Original image')
axes[0].axis('off')

axes[1].imshow(img_8bit, cmap='gray')
axes[1].imshow(masks, cmap='jet', alpha=0.4)
axes[1].set_title(f'Segmentation — {n_cells} cells detected')
axes[1].axis('off')

plt.tight_layout()
plt.savefig('segmentation_result.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved as segmentation_result.png')

# =============================================================================
# STEP 7 — Generate cell ID map for anatomical validation
# -----------------------------------------------------------------------------
# This image lets you identify which numeric ID corresponds to each cell.
# Cross-reference with the CSV to assign cells to tissue types manually.
# =============================================================================

import matplotlib.patheffects as path_effects
from skimage import measure

fig, ax = plt.subplots(figsize=(20, 16))
ax.imshow(img_8bit, cmap='gray')
ax.imshow(masks, cmap='prism', alpha=0.3)

properties = measure.regionprops(masks)

print('Labelling cells with IDs...')
for prop in properties:
    y, x = prop.centroid
    if prop.area > 40:  # skip noise fragments
        txt = ax.text(x, y, str(prop.label),
                      color='white', size=8, weight='bold',
                      ha='center', va='center')
        txt.set_path_effects([path_effects.withStroke(linewidth=1.5, foreground='black')])

ax.set_title('Cell ID map — use this to assign cells to tissue types')
ax.axis('off')
plt.savefig('cell_id_map.png', dpi=300, bbox_inches='tight')
plt.show()

from google.colab import files
files.download('cell_id_map.png')
print('Downloaded: cell_id_map.png')

# =============================================================================
# STEP 8 — Extract quantitative metrics per cell
# -----------------------------------------------------------------------------
# Metrics computed for each segmented cell:
#   Area_px          : cell area in pixels
#   Mean_intensity   : mean raw fluorescence intensity across the whole cell
#   Wall_intensity   : mean intensity of wall pixels (above Otsu threshold)
#   Wall_percentage  : fraction of cell pixels classified as wall (%)
# =============================================================================

import pandas as pd

results = []
for cell_id in range(1, masks.max() + 1):
    cell_mask = (masks == cell_id).astype(np.uint8)
    area = cell_mask.sum()
    if area < 30:
        continue

    raw_pixels  = img_raw[cell_mask == 1]
    pixels_8bit = img_8bit[cell_mask == 1]

    # Otsu threshold applied locally per cell to separate wall from lumen
    otsu_val, _ = cv2.threshold(pixels_8bit, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    wall_pixels = raw_pixels[pixels_8bit > otsu_val]

    results.append({
        'Cell_ID':          cell_id,
        'Area_px':          area,
        'Mean_intensity':   raw_pixels.mean(),
        'Wall_intensity':   wall_pixels.mean() if len(wall_pixels) > 0 else 0,
        'Wall_percentage':  (len(wall_pixels) / len(raw_pixels) * 100) if len(raw_pixels) > 0 else 0
    })

df = pd.DataFrame(results)
print(f'Cells analysed: {len(df)}')
print('\nSummary statistics:')
print(df[['Area_px', 'Mean_intensity', 'Wall_intensity', 'Wall_percentage']].describe().round(2))

# =============================================================================
# STEP 9 — Assign cells to tissue types and compute per-tissue statistics
# -----------------------------------------------------------------------------
# Open cell_id_map.png and identify which IDs belong to each tissue.
# Fill in the lists below with the IDs you select (aim for ≥10 per tissue).
# =============================================================================

# --- Fill these lists after inspecting cell_id_map.png ---
epidermis  = []   # e.g. [5, 12, 23, 31, 44, 58, 67, 73, 81, 95]
cortex     = []   # e.g. [8, 15, 29, 37, 50, 62, 70, 78, 88, 101]
endodermis = []   # e.g. [2, 9, 18, 26, 40, 53, 61, 69, 77, 90]
# ---------------------------------------------------------

df['Tissue'] = 'unassigned'
df.loc[df['Cell_ID'].isin(epidermis),  'Tissue'] = 'Epidermis'
df.loc[df['Cell_ID'].isin(cortex),     'Tissue'] = 'Cortex'
df.loc[df['Cell_ID'].isin(endodermis), 'Tissue'] = 'Endodermis'

assigned = df[df['Tissue'] != 'unassigned']
if len(assigned) > 0:
    print('\nMean metrics by tissue:')
    print(assigned.groupby('Tissue')[
        ['Area_px', 'Mean_intensity', 'Wall_intensity', 'Wall_percentage']
    ].mean().round(2))
else:
    print('⚠️  No cells assigned yet — fill in the tissue ID lists in Step 9.')

# =============================================================================
# STEP 10 — Download results
# =============================================================================

base_name = os.path.splitext(filename)[0]

csv_path  = f'{base_name}_results.csv'
mask_path = f'{base_name}_masks.tif'

df.to_csv(csv_path, index=False)
tfi.imwrite(mask_path, masks.astype(np.uint16))

print('Downloading files...')
files.download(csv_path)
files.download('segmentation_result.png')
files.download(mask_path)
print('✅ Done — check your Downloads folder')
