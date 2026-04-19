"""Compare multiple models on multiple datasets (test split).

This script evaluates checkpoints produced by:
  - CNN.py (SimpleCNN)
  - vision_transformer.py (ViT-B/16)
  - convnext_v2.py (ConvNeXtV2)

It outputs per-dataset per-model metrics:
  - Accuracy
  - Precision / Recall / F1 (weighted)
  - Macro-F1

And saves:
  - confusion matrix images
  - an ACC comparison bar chart per dataset
  - a single CSV file collecting all metrics

Default paths are set for this workspace:
  - PlantVillage: archive/plantvillage dataset/color
  - Rice:        archive1/rice_images
  - Checkpoints: pv_pt/ and r_pt/
"""

from __future__ import annotations

import argparse
import csv
import importlib
import math
import os
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class Metrics:
	acc: float
	precision_weighted: float
	recall_weighted: float
	f1_weighted: float
	f1_macro: float


class SimpleCNN(nn.Module):
	def __init__(self, num_classes: int):
		super().__init__()
		self.features = nn.Sequential(
			nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(64),
			nn.ReLU(inplace=True),
			nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(64),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=2),

			nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(128),
			nn.ReLU(inplace=True),
			nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(128),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=2),

			nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(256),
			nn.ReLU(inplace=True),
			nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(256),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=2),

			nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(512),
			nn.ReLU(inplace=True),
			nn.AdaptiveAvgPool2d((1, 1)),
		)
		self.classifier = nn.Sequential(
			nn.Flatten(),
			nn.Dropout(p=0.3),
			nn.Linear(512, num_classes),
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		x = self.features(x)
		x = self.classifier(x)
		return x


def default_pv_data_root() -> Path:
	here = Path(__file__).resolve().parent
	return here / "archive" / "plantvillage dataset" / "color"


def default_rice_data_root() -> Path:
	here = Path(__file__).resolve().parent
	return here / "archive1" / "rice_images"


def default_ckpt_dir_pv() -> Path:
	here = Path(__file__).resolve().parent
	return here / "pv_pt"


def default_ckpt_dir_rice() -> Path:
	here = Path(__file__).resolve().parent
	return here / "r_pt"


def build_eval_transform(img_size: int) -> transforms.Compose:
	return transforms.Compose(
		[
			transforms.Resize(int(img_size * 1.14)),
			transforms.CenterCrop(img_size),
			transforms.ToTensor(),
			transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
		]
	)


def _safe_torch_load(path: Path, map_location: str = "cpu"):
	try:
		return torch.load(str(path), map_location=map_location, weights_only=False)
	except TypeError:
		# older torch without weights_only
		return torch.load(str(path), map_location=map_location)


def create_vit_b16(num_classes: int) -> nn.Module:
	from torchvision import models
	if hasattr(models, "ViT_B_16_Weights"):
		weights = models.ViT_B_16_Weights.IMAGENET1K_V1
		model = models.vit_b_16(weights=weights)
	else:
		model = models.vit_b_16(pretrained=True)

	# Replace classification head
	if hasattr(model, "heads") and hasattr(model.heads, "head"):
		in_features = model.heads.head.in_features
		model.heads.head = nn.Linear(in_features, num_classes)
	elif hasattr(model, "head"):
		in_features = model.head.in_features
		model.head = nn.Linear(in_features, num_classes)
	else:
		raise RuntimeError("Unsupported ViT head structure")
	return model


def create_convnextv2(num_classes: int, model_name: str = "convnextv2_tiny") -> nn.Module:
	# torchvision first
	try:
		from torchvision import models
		if hasattr(models, model_name):
			ctor = getattr(models, model_name)
			# Handle potential weights enum naming
			weights = None
			weights_enum_name = f"{model_name.upper()}_Weights"
			if hasattr(models, weights_enum_name):
				enum = getattr(models, weights_enum_name)
				# choose first available attribute (typically IMAGENET1K_V1)
				weights = getattr(enum, "IMAGENET1K_V1", None)
			model = ctor(weights=weights) if weights is not None else ctor()
			# Replace classifier head
			if hasattr(model, "classifier") and isinstance(model.classifier, nn.Sequential):
				# torchvision convnext classifier typically ends with Linear
				for i in reversed(range(len(model.classifier))):
					if isinstance(model.classifier[i], nn.Linear):
						in_features = model.classifier[i].in_features
						model.classifier[i] = nn.Linear(in_features, num_classes)
						return model
				raise RuntimeError("Unexpected torchvision ConvNeXtV2 classifier structure")
	except Exception:
		pass

	# timm fallback
	try:
		timm = importlib.import_module("timm")
	except Exception as e:
		raise RuntimeError(
			"ConvNeXtV2 not available in torchvision and timm is not installed. "
			"Install timm with: pip install timm"
		) from e

	# Try a few common timm name variants
	candidates = [model_name, f"{model_name}.fcmae_ft_in1k", f"{model_name}.in1k"]
	last_err: Exception | None = None
	for name in candidates:
		try:
			return timm.create_model(name, pretrained=True, num_classes=num_classes)
		except Exception as e:
			last_err = e
	raise RuntimeError(f"Failed to create convnextv2 via timm, tried {candidates}. Last error: {last_err}")


def create_model(model_key: str, num_classes: int, *, convnext_model: str) -> nn.Module:
	key = model_key.lower()
	if key in {"cnn", "simplecnn"}:
		return SimpleCNN(num_classes=num_classes)
	if key in {"vit", "vision_transformer", "transformer"}:
		return create_vit_b16(num_classes=num_classes)
	if key in {"convnext", "convnextv2", "convnext_v2"}:
		return create_convnextv2(num_classes=num_classes, model_name=convnext_model)
	raise ValueError(f"Unknown model_key: {model_key}")


def confusion_matrix(num_classes: int, y_true: list[int], y_pred: list[int]) -> torch.Tensor:
	cm = torch.zeros((num_classes, num_classes), dtype=torch.int64)
	for t, p in zip(y_true, y_pred, strict=False):
		cm[t, p] += 1
	return cm


def compute_metrics_from_cm(cm: torch.Tensor) -> Metrics:
	cm_f = cm.to(torch.float64)
	total = float(cm_f.sum().item())
	if total <= 0:
		return Metrics(acc=float("nan"), precision_weighted=float("nan"), recall_weighted=float("nan"), f1_weighted=float("nan"), f1_macro=float("nan"))

	tp = torch.diag(cm_f)
	row_sum = cm_f.sum(dim=1)  # support (true)
	col_sum = cm_f.sum(dim=0)

	precision = torch.where(col_sum > 0, tp / col_sum, torch.zeros_like(tp))
	recall = torch.where(row_sum > 0, tp / row_sum, torch.zeros_like(tp))
	f1 = torch.where(
		(precision + recall) > 0,
		2 * precision * recall / (precision + recall),
		torch.zeros_like(tp),
	)

	support = row_sum
	weights = support / total

	acc = float(tp.sum().item() / total)
	precision_w = float((precision * weights).sum().item())
	recall_w = float((recall * weights).sum().item())
	f1_w = float((f1 * weights).sum().item())

	# Macro-F1: average across classes that appear in ground truth
	mask = support > 0
	if mask.any():
		f1_macro = float(f1[mask].mean().item())
	else:
		f1_macro = float("nan")

	return Metrics(acc=acc, precision_weighted=precision_w, recall_weighted=recall_w, f1_weighted=f1_w, f1_macro=f1_macro)


@torch.no_grad()
def predict(
	model: nn.Module,
	loader: DataLoader,
	*,
	device: torch.device,
	use_amp: bool,
) -> tuple[list[int], list[int]]:
	model.eval()
	y_true: list[int] = []
	y_pred: list[int] = []
	for images, targets in loader:
		images = images.to(device, non_blocking=True)
		targets = targets.to(device, non_blocking=True)
		with torch.autocast(device_type=device.type, enabled=use_amp):
			logits = model(images)
			preds = logits.argmax(dim=1)
		y_true.extend(targets.tolist())
		y_pred.extend(preds.tolist())
	return y_true, y_pred


def save_confusion_matrix_image(
	cm: torch.Tensor,
	class_names: list[str],
	*,
	title: str,
	out_path: Path,
) -> None:
	import matplotlib
	matplotlib.use("Agg")
	import matplotlib.pyplot as plt

	cm_np = cm.cpu().numpy()
	n = cm_np.shape[0]
	fig_w = max(8.0, min(24.0, 0.35 * n))
	fig_h = max(7.0, min(20.0, 0.35 * n))
	plt.figure(figsize=(fig_w, fig_h))
	plt.imshow(cm_np, interpolation="nearest", cmap="Blues")
	plt.title(title)
	plt.colorbar(fraction=0.046, pad=0.04)

	tick_step = 1
	if n > 40:
		tick_step = 2
	if n > 80:
		tick_step = 4

	ticks = list(range(0, n, tick_step))
	plt.xticks(ticks, [class_names[i] for i in ticks], rotation=90, fontsize=7)
	plt.yticks(ticks, [class_names[i] for i in ticks], fontsize=7)
	plt.ylabel("True")
	plt.xlabel("Pred")

	plt.tight_layout()
	out_path.parent.mkdir(parents=True, exist_ok=True)
	plt.savefig(str(out_path), dpi=220)
	plt.close()


def save_acc_bar_chart(
	rows: list[dict],
	*,
	dataset_name: str,
	out_path: Path,
) -> None:
	import matplotlib
	matplotlib.use("Agg")
	import matplotlib.pyplot as plt

	models = [r["model"] for r in rows]
	accs = [float(r["acc"]) for r in rows]
	plt.figure(figsize=(8, 4.5))
	bars = plt.bar(models, accs)
	plt.ylim(0.0, 1.0)
	plt.ylabel("Accuracy")
	plt.title(f"Accuracy comparison ({dataset_name})")
	plt.grid(axis="y", linestyle="--", alpha=0.35)
	for b, v in zip(bars, accs, strict=False):
		plt.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v*100:.2f}%", ha="center", va="bottom", fontsize=9)
	plt.tight_layout()
	out_path.parent.mkdir(parents=True, exist_ok=True)
	plt.savefig(str(out_path), dpi=220)
	plt.close()


def _best_match_ckpt(ckpt_dir: Path, pattern: str) -> Path:
	matches = sorted(ckpt_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
	if not matches:
		raise FileNotFoundError(f"No checkpoint matched {pattern} under: {ckpt_dir}")
	return matches[0]


def evaluate_one(
	*,
	dataset_name: str,
	data_root: Path,
	ckpt_path: Path,
	model_key: str,
	batch_size: int,
	num_workers: int,
	device: torch.device,
	use_amp: bool,
	convnext_model: str,
	out_dir: Path,
) -> dict:
	data_root = data_root.expanduser().resolve()
	ckpt_path = ckpt_path.expanduser().resolve()

	test_dir = data_root / "test"
	if not test_dir.is_dir():
		raise FileNotFoundError(f"Missing test directory: {test_dir}")

	ckpt = _safe_torch_load(ckpt_path, map_location="cpu")
	if not isinstance(ckpt, dict) or "model_state" not in ckpt or "class_to_idx" not in ckpt:
		raise RuntimeError(f"Unexpected checkpoint format: {ckpt_path}")

	img_size = int(ckpt.get("img_size", 224))
	class_to_idx = ckpt["class_to_idx"]
	if not isinstance(class_to_idx, dict) or not class_to_idx:
		raise RuntimeError(f"Invalid class_to_idx in checkpoint: {ckpt_path}")

	test_tf = build_eval_transform(img_size)
	test_ds = datasets.ImageFolder(str(test_dir), transform=test_tf)
	if test_ds.class_to_idx != class_to_idx:
		raise RuntimeError(
			"Class mapping mismatch between checkpoint and dataset. "
			f"ckpt={ckpt_path} data_root={data_root}"
		)

	# Windows stability: allow user to set workers, but enforce >=0.
	nw = max(0, int(num_workers))
	loader_kwargs = {
		"num_workers": nw,
		"pin_memory": True,
		"persistent_workers": (nw > 0),
	}
	if nw > 0:
		loader_kwargs["prefetch_factor"] = 2
	loader = DataLoader(test_ds, batch_size=int(batch_size), shuffle=False, **loader_kwargs)

	num_classes = len(test_ds.classes)
	model = create_model(model_key, num_classes, convnext_model=convnext_model)
	model.load_state_dict(ckpt["model_state"], strict=True)
	model.to(device)

	y_true, y_pred = predict(model, loader, device=device, use_amp=use_amp)
	cm = confusion_matrix(num_classes, y_true, y_pred)
	metrics = compute_metrics_from_cm(cm)

	# Save confusion matrix image
	cm_path = out_dir / dataset_name / f"{model_key}_confusion.png"
	save_confusion_matrix_image(cm, test_ds.classes, title=f"{dataset_name} | {model_key}", out_path=cm_path)

	return {
		"dataset": dataset_name,
		"model": model_key,
		"ckpt": str(ckpt_path),
		"img_size": img_size,
		"num_classes": num_classes,
		"num_samples": len(test_ds),
		"acc": metrics.acc,
		"precision": metrics.precision_weighted,
		"recall": metrics.recall_weighted,
		"f1": metrics.f1_weighted,
		"macro_f1": metrics.f1_macro,
	}


def write_metrics_csv(rows: list[dict], out_path: Path) -> None:
	fieldnames = [
		"dataset",
		"model",
		"acc",
		"precision",
		"recall",
		"f1",
		"macro_f1",
		"num_classes",
		"num_samples",
		"img_size",
		"ckpt",
	]
	out_path.parent.mkdir(parents=True, exist_ok=True)
	with out_path.open("w", newline="", encoding="utf-8") as f:
		w = csv.DictWriter(f, fieldnames=fieldnames)
		w.writeheader()
		for r in rows:
			w.writerow(r)


def main(argv: list[str] | None = None) -> int:
	parser = argparse.ArgumentParser(description="Compare CNN/ViT/ConvNeXtV2 checkpoints on two datasets")
	parser.add_argument("--pv_data_root", type=Path, default=default_pv_data_root(), help="PlantVillage data root containing train/valid/test")
	parser.add_argument("--rice_data_root", type=Path, default=default_rice_data_root(), help="Rice data root containing train/valid/test")
	parser.add_argument("--pv_ckpt_dir", type=Path, default=default_ckpt_dir_pv(), help="Directory containing PV checkpoints")
	parser.add_argument("--rice_ckpt_dir", type=Path, default=default_ckpt_dir_rice(), help="Directory containing Rice checkpoints")
	parser.add_argument("--batch_size", type=int, default=64, help="Test batch size")
	parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers (Windows建议0更稳)")
	parser.add_argument("--no_amp", action="store_true", help="Disable AMP during inference")
	parser.add_argument("--convnext_model", type=str, default="convnextv2_tiny", help="ConvNeXtV2 variant name")
	parser.add_argument("--output_dir", type=Path, default=Path("compare_outputs") / "pt_compare", help="Directory to save CSV and images")
	parser.add_argument("--device", type=str, default="cuda", help="Device: cuda")

	# Allow overriding exact checkpoint paths
	parser.add_argument("--pv_cnn", type=Path, default=None, help="PlantVillage CNN checkpoint path")
	parser.add_argument("--pv_vit", type=Path, default=None, help="PlantVillage ViT checkpoint path")
	parser.add_argument("--pv_convnext", type=Path, default=None, help="PlantVillage ConvNeXtV2 checkpoint path")
	parser.add_argument("--rice_cnn", type=Path, default=None, help="Rice CNN checkpoint path")
	parser.add_argument("--rice_vit", type=Path, default=None, help="Rice ViT checkpoint path")
	parser.add_argument("--rice_convnext", type=Path, default=None, help="Rice ConvNeXtV2 checkpoint path")

	args = parser.parse_args(argv)

	if args.device.lower() != "cuda":
		raise RuntimeError("This script is CUDA-only for consistency. Use --device cuda.")
	if not torch.cuda.is_available():
		raise RuntimeError("CUDA is not available. Please install CUDA-enabled PyTorch and NVIDIA drivers.")
	device = torch.device("cuda")
	use_amp = not bool(args.no_amp)

	out_dir = Path(args.output_dir).expanduser().resolve()
	out_dir.mkdir(parents=True, exist_ok=True)

	# Resolve checkpoints
	pv_dir = Path(args.pv_ckpt_dir).expanduser().resolve()
	r_dir = Path(args.rice_ckpt_dir).expanduser().resolve()

	pv_cnn = Path(args.pv_cnn) if args.pv_cnn is not None else _best_match_ckpt(pv_dir, "*cnn*.pt")
	pv_vit = Path(args.pv_vit) if args.pv_vit is not None else _best_match_ckpt(pv_dir, "*vit*.pt")
	pv_conv = Path(args.pv_convnext) if args.pv_convnext is not None else _best_match_ckpt(pv_dir, "*convnext*.pt")

	r_cnn = Path(args.rice_cnn) if args.rice_cnn is not None else _best_match_ckpt(r_dir, "*cnn*.pt")
	r_vit = Path(args.rice_vit) if args.rice_vit is not None else _best_match_ckpt(r_dir, "*vit*.pt")
	r_conv = Path(args.rice_convnext) if args.rice_convnext is not None else _best_match_ckpt(r_dir, "*convnext*.pt")

	rows: list[dict] = []

	# Evaluate PV
	for model_key, ckpt_path in [("cnn", pv_cnn), ("vit", pv_vit), ("convnextv2", pv_conv)]:
		row = evaluate_one(
			dataset_name="PlantVillage",
			data_root=Path(args.pv_data_root),
			ckpt_path=ckpt_path,
			model_key=model_key,
			batch_size=int(args.batch_size),
			num_workers=int(args.num_workers),
			device=device,
			use_amp=use_amp,
			convnext_model=str(args.convnext_model),
			out_dir=out_dir,
		)
		rows.append(row)
		print(
			f"[PlantVillage] {model_key}: acc={row['acc']*100:.2f}% "
			f"F1={row['f1']*100:.2f}% Macro-F1={row['macro_f1']*100:.2f}%"
		)

	# Evaluate Rice
	for model_key, ckpt_path in [("cnn", r_cnn), ("vit", r_vit), ("convnextv2", r_conv)]:
		row = evaluate_one(
			dataset_name="Rice",
			data_root=Path(args.rice_data_root),
			ckpt_path=ckpt_path,
			model_key=model_key,
			batch_size=int(args.batch_size),
			num_workers=int(args.num_workers),
			device=device,
			use_amp=use_amp,
			convnext_model=str(args.convnext_model),
			out_dir=out_dir,
		)
		rows.append(row)
		print(
			f"[Rice] {model_key}: acc={row['acc']*100:.2f}% "
			f"F1={row['f1']*100:.2f}% Macro-F1={row['macro_f1']*100:.2f}%"
		)

	# Write combined CSV
	csv_path = out_dir / "metrics.csv"
	write_metrics_csv(rows, csv_path)

	# ACC bar charts per dataset
	pv_rows = [r for r in rows if r["dataset"] == "PlantVillage"]
	r_rows = [r for r in rows if r["dataset"] == "Rice"]
	save_acc_bar_chart(pv_rows, dataset_name="PlantVillage", out_path=out_dir / "PlantVillage" / "acc_compare.png")
	save_acc_bar_chart(r_rows, dataset_name="Rice", out_path=out_dir / "Rice" / "acc_compare.png")

	print(f"\nSaved metrics CSV: {csv_path}")
	print(f"Saved images under: {out_dir}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())

