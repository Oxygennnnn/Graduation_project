# Graduation_project
A Comparative Study of Deep Learning Models for Crop Disease Classification

This repository provides a lightweight and reproducible pipeline to compare three deep learning models for crop disease classification:

- CNN baseline (SimpleCNN)
- Vision Transformer (ViT-B/16)
- ConvNeXtV2 (torchvision if available, otherwise timm fallback)

It supports training and testing on two datasets (e.g., PlantVillage and Rice) using a unified ImageFolder-style directory layout.

---

## Repository Overview

Main scripts:

- CNN training: `CNN.py`
- ViT training: `vision_transformer.py`
- ConvNeXtV2 training: `convnext_v2.py`
- Dataset split utility (create train/valid/test): `data_division.py`
- Evaluate & compare checkpoints (CSV + figures): `model_compare.py`
- Optional duplicate/near-duplicate check across splits: `near_duplicate_check.py`

---

## Requirements

### Python

- Python **>= 3.10**
	- The code uses modern type syntax like `list[str] | None`, which requires Python 3.10+.

### GPU / CUDA (Important)

All training and evaluation scripts in this repo are **CUDA-only** for consistency.
If CUDA is not available, scripts will raise an error.

Recommended check:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

---

## Installation

Create an environment (optional but recommended), then install dependencies:

```bash
pip install torch torchvision pillow matplotlib
```

ConvNeXtV2 model creation:

- If your torchvision version does not provide ConvNeXtV2, install timm:

```bash
pip install timm
```

Note:

- For CUDA-enabled PyTorch on Windows, install a CUDA build of PyTorch that matches your GPU driver.
	(Follow the official PyTorch installation instructions for your CUDA version.)

---

## Dataset Preparation

### Required dataset layout (ImageFolder + splits)

All training scripts expect this structure:

```
<DATA_ROOT>/
	train/
		<class_1>/*.jpg
		<class_2>/*.jpg
		...
	valid/
		<class_1>/*.jpg
		<class_2>/*.jpg
		...
	test/
		<class_1>/*.jpg
		<class_2>/*.jpg
		...
```

### Split a dataset into train/valid/test (7:2:1)

Use `data_division.py`.

Case A: Your dataset is “flat” (class folders directly under root):

```
<DATA_ROOT>/
	<class_1>/*.jpg
	<class_2>/*.jpg
```

Run:

```bash
python data_division.py --data_root "PATH_TO_DATA_ROOT"
```

By default it **moves** images into train/valid/test.
If you want to keep the original files in place, use `--copy`:

```bash
python data_division.py --data_root "PATH_TO_DATA_ROOT" --copy
```

If `valid/` or `test/` is already non-empty, it will stop for safety.
Use `--force` only if you are sure:

```bash
python data_division.py --data_root "PATH_TO_DATA_ROOT" --force
```

---

## Recommended Local Folder Layout (This Repo)

The training scripts default to a PlantVillage path under `archive1/...`.
For easiest reproduction, place datasets under the repo like this:

PlantVillage (example):

```
archive1/
	New Plant Diseases Dataset(Augmented)/
		New Plant Diseases Dataset(Augmented)/
			train/...
			valid/...
			test/...
```

Rice (example):

```
archive1/
	rice_images/
		train/...
		valid/...
		test/...
```

If your datasets are elsewhere, you can still run everything by passing `--data_root` / `--pv_data_root` / `--rice_data_root`.

---

## Training

### Windows note (important)

On Windows, DataLoader multiprocessing can sometimes hang. If you encounter issues, set:

- `--num_workers 0`

### 1) Train on PlantVillage

Create checkpoint directory:

```bash
mkdir pv_pt
```

CNN:

```bat
python CNN.py ^
	--data_root "archive1/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)" ^
	--epochs 30 --batch_size 32 --num_workers 0 ^
	--output "pv_pt/cnn_best.pt"
```

ViT:

```bat
python vision_transformer.py ^
	--data_root "archive1/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)" ^
	--epochs 30 --batch_size 32 --num_workers 0 ^
	--output "pv_pt/vit_best.pt"
```

ConvNeXtV2:

```bat
python convnext_v2.py ^
	--data_root "archive1/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)" ^
	--model convnextv2_tiny ^
	--epochs 30 --batch_size 32 --num_workers 0 ^
	--output "pv_pt/convnextv2_best.pt"
```

What training scripts do:

- Train using `train` split
- Track validation loss on `valid` split
- Save/overwrite the best checkpoint when validation improves
- Load best checkpoint and report TEST accuracy on `test` split at the end

Common useful flags (all training scripts):

- `--lr 0.0` means auto LR (LR range test). You can override with `--lr 3e-4`
- `--no_lr_find` to disable LR range test
- `--no_amp` to disable mixed precision
- `--patience` early stopping patience (default 6)

### 2) Train on Rice

Create checkpoint directory:

```bash
mkdir r_pt
```

CNN:

```bat
python CNN.py ^
	--data_root "archive1/rice_images" ^
	--epochs 30 --batch_size 32 --num_workers 0 ^
	--output "r_pt/cnn_best.pt"
```

ViT:

```bat
python vision_transformer.py ^
	--data_root "archive1/rice_images" ^
	--epochs 30 --batch_size 32 --num_workers 0 ^
	--output "r_pt/vit_best.pt"
```

ConvNeXtV2:

```bat
python convnext_v2.py ^
	--data_root "archive1/rice_images" ^
	--model convnextv2_tiny ^
	--epochs 30 --batch_size 32 --num_workers 0 ^
	--output "r_pt/convnextv2_best.pt"
```

---

## Evaluation & Comparison (CSV + Figures)

Run `model_compare.py` to evaluate all checkpoints on the test splits and produce:

- `metrics.csv` (accuracy, precision, recall, F1, macro-F1)
- confusion matrix images per dataset
- an accuracy comparison bar chart per dataset

Important:

- `model_compare.py` has its own default dataset paths, which may not match the training defaults.
	To avoid mismatch, it is recommended to pass dataset paths explicitly.

Example:

```bat
python model_compare.py ^
	--pv_data_root "archive1/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)" ^
	--rice_data_root "archive1/rice_images" ^
	--pv_ckpt_dir "pv_pt" ^
	--rice_ckpt_dir "r_pt" ^
	--batch_size 64 --num_workers 0 ^
	--output_dir "compare_outputs/pt_compare"
```

If your checkpoint filenames do not match the patterns, you can override them explicitly:

- `--pv_cnn`, `--pv_vit`, `--pv_convnext`
- `--rice_cnn`, `--rice_vit`, `--rice_convnext`

---

## Outputs

### Training outputs

Each training script writes one checkpoint (default names shown):

- `cnn_best.pt`
- `vit_best.pt`
- `convnextv2_best.pt`

Recommended locations:

- PlantVillage: `pv_pt/*.pt`
- Rice: `r_pt/*.pt`

Checkpoint content (dict):

- `model_state`, `class_to_idx`, `img_size`, `epoch`, `val_loss`

### Comparison outputs

By default (or if you use the example command), outputs go to:

```
compare_outputs/pt_compare/
	metrics.csv
	PlantVillage/
		cnn_confusion.png
		vit_confusion.png
		convnextv2_confusion.png
		acc_compare.png
	Rice/
		cnn_confusion.png
		vit_confusion.png
		convnextv2_confusion.png
		acc_compare.png
```

If you ran older versions previously, you may see slightly different filenames.

---

## Optional: Duplicate / Near-Duplicate Check

Augmented datasets may leak near-duplicates across train/valid/test.
This script reports collisions using perceptual dHash.

Run:

```bat
python near_duplicate_check.py ^
	--data_root "archive1/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)" ^
	--output "near_duplicate_report.json"
```

Optional near-duplicate search (more expensive):

```bat
python near_duplicate_check.py ^
	--data_root "archive1/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)" ^
	--bucket_prefix_bits 16 ^
	--near_hamming 3 ^
	--output "near_duplicate_report.json"
```

---

## Troubleshooting

### 1) CUDA is not available

Symptoms:

- Script exits with `CUDA is not available...`

Fix:

- Install CUDA-enabled PyTorch and ensure NVIDIA drivers are installed.
- Verify: `python -c "import torch; print(torch.cuda.is_available())"`

### 2) Windows DataLoader hangs / very slow

Fix:

- Use `--num_workers 0` for training and comparison scripts.

### 3) Out of memory (OOM)

Fix:

- Reduce `--batch_size` (e.g., 16 or 8)
- Optionally use `--no_amp` if AMP causes instability (rare)

### 4) ConvNeXtV2 model not found

Fix:

- `pip install timm`
- Or upgrade torchvision to a version that includes ConvNeXtV2.

### 5) Class mismatch across splits / checkpoint mismatch

Symptoms:

- Error about class mapping mismatch between checkpoint and dataset

Fix:

- Ensure train/valid/test have identical class folder names.
- Re-run `data_division.py` on a clean dataset layout if needed.

---

## Notes

- Seeds can be controlled using `--seed` for reproducibility.
- This repo does not include datasets or pretrained checkpoints by default.
	You need to download datasets and train models locally.

---

## License

Add your license here.

## Author

Add author info here.
