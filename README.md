# Master-s-thesis
Confocal Microscopy Cell Segmentation
# Root Cell Wall Image Analysis Pipeline

Quantitative image analysis pipeline for confocal microscopy images of plant root cross-sections. Developed as part of a master's thesis investigating how salt stress alters root cell wall structure and composition in halophytic Brassicaceae species (*Eutrema salsugineum* and *Schrenkiella parvula*) compared to *Arabidopsis thaliana*.

**Institution:** AgroParisTech (Université Paris-Saclay) / École Polytechnique Fédérale de Lausanne, Plant Adaptation Laboratory  
**Supervisor:** Dr. Priya Ramakrishna

---

## What this pipeline does

1. Loads a confocal microscopy TIFF image (grayscale, exported from Fiji/ImageJ)
2. Segments individual cells using **Cellpose** deep learning models
3. Generates a labelled cell ID map for anatomical tissue assignment
4. Extracts per-cell quantitative metrics:
   - Cell area (px)
   - Mean fluorescence intensity
   - Cell wall intensity (pixels above Otsu threshold)
   - Cell wall percentage (fraction of cell area occupied by wall signal)
5. Computes mean metrics per tissue type (epidermis, cortex, endodermis)
6. Exports results as CSV and segmentation masks as TIFF

---

## Biological context

Images were acquired by confocal microscopy using:
- **Calcofluor White** staining (cell wall / cellulose signal)
- **Basic Fuchsin** staining (lignin signal)

Raw images are in Olympus `.oir` format and exported as single-channel grayscale TIFFs for analysis. The pipeline was applied to root tip cross-sections spanning the meristematic and elongation zones.

---

## Requirements

This pipeline runs on **Google Colab** (free, no local installation required). A Google account is sufficient.

The following packages are installed automatically by the notebook:
- `cellpose >= 3.0`
- `numpy >= 1.24, < 2.1`
- `tifffile`
- `scipy`
- `pandas`
- `matplotlib`
- `opencv-python-headless`
- `scikit-image`

> **Note:** The pipeline requires a GPU for reasonable execution speed. In Colab, go to `Runtime > Change runtime type > T4 GPU` before running.

---

## How to use

1. Open [Google Colab](https://colab.research.google.com)
2. Upload `segmentacion_cellpose.py` via `File > Upload notebook`
3. Enable GPU: `Runtime > Change runtime type > T4 GPU`
4. Upload your TIFF image to Google Drive
5. Run cells in order (▶ button), following the instructions in each step
6. In **Step 9**, open the downloaded `cell_id_map.png` and manually assign cell IDs to tissue types (epidermis, cortex, endodermis)
7. Download the CSV results and segmentation masks from **Step 10**

---

## Key parameters (Step 5)

| Parameter | Default | Effect |
|---|---|---|
| `diameter` | 30 px | Estimated cell diameter. Increase if fragments detected; decrease if cells missed |
| `model_type` | `cyto3` | Cellpose model. `cyto3` recommended for plant cells |
| `flow_threshold` | 0.4 | Higher = stricter, fewer cells |
| `cellprob_threshold` | 0.0 | Higher = only high-confidence detections |

---

## Output files

| File | Description |
|---|---|
| `*_results.csv` | Per-cell metrics table |
| `segmentation_result.png` | Visual overlay of segmentation masks |
| `cell_id_map.png` | High-resolution map with cell IDs labelled |
| `*_masks.tif` | Segmentation masks as 16-bit TIFF (one integer per cell) |

---

## Statistical considerations

A minimum of **10 cells per tissue type per image** is recommended for descriptive statistics. For between-group comparisons (e.g. control vs. salt-treated, or across species), at least **3 independent biological replicates** (separate roots) are required.

---

## Reference

Cellpose: Stringer, C., Wang, T., Michaelos, M., & Pachitariu, M. (2021). Cellpose: a generalist algorithm for cellular segmentation. *Nature Methods*, 18, 100–106. https://doi.org/10.1038/s41592-020-01018-x
