"""Vision Transformer training script (train/valid/test).

This script expects a dataset root containing:

	<data_root>/train/<class_name>/*
	<data_root>/valid/<class_name>/*
	<data_root>/test/<class_name>/*

It fine-tunes a pretrained ViT (torchvision) for image classification.

Features:
  - Strong-but-safe image augmentation for classification.
  - Auto LR selection via a lightweight LR range test (optional).
  - Early stopping with configurable patience.
  - Concise terminal logs.

Example:
  python vision_transformer.py --data_root "D:\\path\\to\\New Plant Diseases Dataset(Augmented)\\New Plant Diseases Dataset(Augmented)"
"""

from __future__ import annotations

import argparse
import copy
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class EpochMetrics:
	loss: float
	acc: float


def seed_everything(seed: int) -> None:
	random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)


def default_data_root() -> Path:
	"""Default to the common dataset location relative to this workspace."""
	here = Path(__file__).resolve().parent
	# data_root is the parent directory that contains train/valid/test
	# (which is the parent of the 'train' folder used in data_division.py)
	return (
		here
		/ "archive1"
		/ "New Plant Diseases Dataset(Augmented)"
		/ "New Plant Diseases Dataset(Augmented)"
	)


def build_transforms(img_size: int) -> tuple[transforms.Compose, transforms.Compose]:
	"""Train/val transforms.

	Keeps them compatible across torchvision versions.
	"""

	# Augmentations tuned for disease classification: modest geometry + color jitter.
	# Prefer RandAugment if available; fallback to AutoAugment.
	aug_list = []
	if hasattr(transforms, "RandAugment"):
		aug_list.append(transforms.RandAugment(num_ops=2, magnitude=9))
	elif hasattr(transforms, "AutoAugment"):
		policy = getattr(transforms.AutoAugmentPolicy, "IMAGENET", None)
		if policy is not None:
			aug_list.append(transforms.AutoAugment(policy=policy))

	train_tf = transforms.Compose(
		[
			transforms.RandomResizedCrop(img_size, scale=(0.75, 1.0), ratio=(0.9, 1.1)),
			transforms.RandomHorizontalFlip(p=0.5),
			transforms.RandomApply([transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)], p=0.6),
			*aug_list,
			transforms.ToTensor(),
			transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
			transforms.RandomErasing(p=0.25, scale=(0.02, 0.12), ratio=(0.3, 3.3)),
		]
	)

	eval_tf = transforms.Compose(
		[
			transforms.Resize(int(img_size * 1.14)),
			transforms.CenterCrop(img_size),
			transforms.ToTensor(),
			transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
		]
	)
	return train_tf, eval_tf


def accuracy_top1(logits: torch.Tensor, targets: torch.Tensor) -> float:
	preds = logits.argmax(dim=1)
	return (preds == targets).float().mean().item()


@torch.no_grad()
def evaluate(
	model: nn.Module,
	loader: DataLoader,
	device: torch.device,
	criterion: nn.Module,
	use_amp: bool,
) -> EpochMetrics:
	model.eval()
	total_loss = 0.0
	total_correct = 0
	total = 0

	for images, targets in loader:
		images = images.to(device, non_blocking=True)
		targets = targets.to(device, non_blocking=True)
		with torch.autocast(device_type=device.type, enabled=use_amp):
			logits = model(images)
			loss = criterion(logits, targets)
		batch = targets.size(0)
		total_loss += loss.item() * batch
		total_correct += (logits.argmax(dim=1) == targets).sum().item()
		total += batch

	if total == 0:
		return EpochMetrics(loss=float("nan"), acc=float("nan"))
	return EpochMetrics(loss=total_loss / total, acc=total_correct / total)


def train_one_epoch(
	model: nn.Module,
	loader: DataLoader,
	device: torch.device,
	criterion: nn.Module,
	optimizer: torch.optim.Optimizer,
	scaler: torch.cuda.amp.GradScaler | None,
	use_amp: bool,
) -> EpochMetrics:
	model.train()
	total_loss = 0.0
	total_correct = 0
	total = 0

	for images, targets in loader:
		images = images.to(device, non_blocking=True)
		targets = targets.to(device, non_blocking=True)

		optimizer.zero_grad(set_to_none=True)
		with torch.autocast(device_type=device.type, enabled=use_amp):
			logits = model(images)
			loss = criterion(logits, targets)

		if scaler is not None and use_amp:
			scaler.scale(loss).backward()
			scaler.step(optimizer)
			scaler.update()
		else:
			loss.backward()
			optimizer.step()

		batch = targets.size(0)
		total_loss += loss.item() * batch
		total_correct += (logits.argmax(dim=1) == targets).sum().item()
		total += batch

	if total == 0:
		return EpochMetrics(loss=float("nan"), acc=float("nan"))
	return EpochMetrics(loss=total_loss / total, acc=total_correct / total)


def lr_range_test(
	model: nn.Module,
	loader: DataLoader,
	device: torch.device,
	criterion: nn.Module,
	optimizer_ctor,
	start_lr: float = 1e-6,
	end_lr: float = 1e-2,
	num_iter: int = 120,
	use_amp: bool = True,
) -> float:
	"""A lightweight LR finder.

	Increases LR exponentially from start_lr to end_lr over num_iter steps,
	tracks loss, and returns a suggested LR.
	"""

	if num_iter <= 10:
		raise ValueError("num_iter should be > 10")
	if start_lr <= 0 or end_lr <= 0 or end_lr <= start_lr:
		raise ValueError("invalid LR range")

	model_state = copy.deepcopy(model.state_dict())

	optimizer = optimizer_ctor(model.parameters(), lr=start_lr)
	scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and device.type == "cuda"))

	# exponential schedule
	mult = (end_lr / start_lr) ** (1 / (num_iter - 1))
	lr = start_lr
	for pg in optimizer.param_groups:
		pg["lr"] = lr

	losses: list[float] = []
	lrs: list[float] = []
	best_loss = float("inf")

	model.train()
	it = iter(loader)
	for i in range(num_iter):
		try:
			images, targets = next(it)
		except StopIteration:
			it = iter(loader)
			images, targets = next(it)

		images = images.to(device, non_blocking=True)
		targets = targets.to(device, non_blocking=True)

		optimizer.zero_grad(set_to_none=True)
		with torch.autocast(device_type=device.type, enabled=use_amp):
			logits = model(images)
			loss = criterion(logits, targets)

		if torch.isnan(loss) or torch.isinf(loss):
			break

		if scaler.is_enabled():
			scaler.scale(loss).backward()
			scaler.step(optimizer)
			scaler.update()
		else:
			loss.backward()
			optimizer.step()

		loss_val = loss.item()
		losses.append(loss_val)
		lrs.append(lr)
		best_loss = min(best_loss, loss_val)

		# stop if loss explodes too much
		if loss_val > best_loss * 6 and i > 15:
			break

		lr *= mult
		for pg in optimizer.param_groups:
			pg["lr"] = lr

	# Restore model
	model.load_state_dict(model_state)

	if len(losses) < 20:
		# fallback
		return 3e-4

	# Smooth losses with simple EMA
	beta = 0.9
	avg = 0.0
	smoothed: list[float] = []
	for j, v in enumerate(losses):
		avg = beta * avg + (1 - beta) * v
		corr = 1 - beta ** (j + 1)
		smoothed.append(avg / corr)

	min_idx = int(min(range(len(smoothed)), key=lambda k: smoothed[k]))
	suggested = lrs[min_idx] / 10.0
	suggested = float(max(start_lr, min(suggested, end_lr)))
	return suggested


def create_model(num_classes: int) -> nn.Module:
	from torchvision import models

	# Handle torchvision version differences.
	weights = None
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
		raise RuntimeError("Unsupported ViT model head structure.")
	return model


def main(argv: list[str] | None = None) -> int:
	parser = argparse.ArgumentParser(description="Fine-tune Vision Transformer on ImageFolder dataset")
	parser.add_argument("--data_root", type=Path, default=default_data_root(), help="Folder containing train/valid/test")
	parser.add_argument("--img_size", type=int, default=224, help="Input image size")
	parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
	parser.add_argument("--epochs", type=int, default=30, help="Max epochs")
	parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
	parser.add_argument("--seed", type=int, default=42, help="Random seed")
	parser.add_argument("--patience", type=int, default=6, help="Early stopping patience (epochs)")
	parser.add_argument("--lr", type=float, default=0.0, help="Learning rate override; 0 means auto")
	parser.add_argument("--weight_decay", type=float, default=0.05, help="AdamW weight decay")
	parser.add_argument("--label_smoothing", type=float, default=0.1, help="CrossEntropy label smoothing")
	parser.add_argument("--no_amp", action="store_true", help="Disable AMP mixed precision")
	parser.add_argument("--no_lr_find", action="store_true", help="Disable LR range test")
	parser.add_argument("--lr_find_iters", type=int, default=120, help="LR range test iterations")
	parser.add_argument("--output", type=Path, default=Path("vit_best.pt"), help="Path to save best model")

	args = parser.parse_args(argv)
	seed_everything(args.seed)

	data_root = args.data_root.expanduser().resolve()
	train_dir = data_root / "train"
	valid_dir = data_root / "valid"
	test_dir = data_root / "test"

	if not train_dir.is_dir() or not valid_dir.is_dir() or not test_dir.is_dir():
		raise FileNotFoundError(
			f"Expected train/valid/test under data_root. Missing one of: {train_dir}, {valid_dir}, {test_dir}"
		)

	if not torch.cuda.is_available():
		raise RuntimeError("CUDA is not available. Please install CUDA-enabled PyTorch and NVIDIA drivers.")
	device = torch.device("cuda")
	use_amp = not args.no_amp

	# CUDA performance knobs (safe defaults)
	torch.backends.cudnn.benchmark = True
	if hasattr(torch.backends.cudnn, "allow_tf32"):
		torch.backends.cudnn.allow_tf32 = True
	if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
		torch.backends.cuda.matmul.allow_tf32 = True
	try:
		torch.set_float32_matmul_precision("high")
	except Exception:
		pass

	train_tf, eval_tf = build_transforms(args.img_size)
	train_ds = datasets.ImageFolder(str(train_dir), transform=train_tf)
	valid_ds = datasets.ImageFolder(str(valid_dir), transform=eval_tf)
	test_ds = datasets.ImageFolder(str(test_dir), transform=eval_tf)

	# Ensure class mapping is consistent
	if valid_ds.class_to_idx != train_ds.class_to_idx or test_ds.class_to_idx != train_ds.class_to_idx:
		raise RuntimeError("Class folders mismatch across train/valid/test. Ensure same subfolder names.")

	num_classes = len(train_ds.classes)
	if num_classes <= 1:
		raise RuntimeError(f"Need at least 2 classes, got: {num_classes}")

	# Data decoding/augment still runs on CPU (torchvision), but we pin memory to speed CPU->GPU transfer.
	pin = True
	loader_kwargs = {
		"num_workers": args.num_workers,
		"pin_memory": pin,
		"persistent_workers": (args.num_workers > 0),
	}
	if args.num_workers > 0:
		loader_kwargs["prefetch_factor"] = 2
	train_loader = DataLoader(
		train_ds,
		batch_size=args.batch_size,
		shuffle=True,
		**loader_kwargs,
	)
	valid_loader = DataLoader(
		valid_ds,
		batch_size=args.batch_size,
		shuffle=False,
		**loader_kwargs,
	)
	test_loader = DataLoader(
		test_ds,
		batch_size=args.batch_size,
		shuffle=False,
		**loader_kwargs,
	)

	model = create_model(num_classes=num_classes).to(device)
	criterion = nn.CrossEntropyLoss(label_smoothing=float(args.label_smoothing))

	def optimizer_ctor(params, lr: float):
		return torch.optim.AdamW(params, lr=lr, weight_decay=float(args.weight_decay))

	# Auto LR
	lr = float(args.lr)
	if lr <= 0:
		if args.no_lr_find:
			lr = 3e-4
			print(f"Using default lr={lr:.2e} (lr_find disabled)")
		else:
			t0 = time.time()
			lr = lr_range_test(
				model=model,
				loader=train_loader,
				device=device,
				criterion=criterion,
				optimizer_ctor=optimizer_ctor,
				start_lr=1e-6,
				end_lr=3e-3,
				num_iter=int(args.lr_find_iters),
				use_amp=use_amp,
			)
			print(f"Auto LR suggested: lr={lr:.2e} (range-test {time.time()-t0:.1f}s)")
	else:
		print(f"Using user lr={lr:.2e}")

	optimizer = optimizer_ctor(model.parameters(), lr=lr)
	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
	scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

	out_path = args.output
	if not out_path.is_absolute():
		out_path = (Path(__file__).resolve().parent / out_path).resolve()
	out_path.parent.mkdir(parents=True, exist_ok=True)

	best_val_loss = float("inf")
	best_epoch = -1
	bad_epochs = 0

	def save_best_checkpoint(epoch: int, val_loss: float) -> None:
		"""Save best weights to a single file (overwrite/refresh when improved)."""
		tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
		payload = {
			"model_state": model.state_dict(),
			"class_to_idx": train_ds.class_to_idx,
			"img_size": args.img_size,
			"epoch": int(epoch),
			"val_loss": float(val_loss),
		}
		torch.save(payload, str(tmp_path))
		os.replace(str(tmp_path), str(out_path))

	print(
		f"Device=cuda | classes={num_classes} | train={len(train_ds)} valid={len(valid_ds)} test={len(test_ds)} | lr={lr:.2e} | amp={'on' if use_amp else 'off'}"
	)

	for epoch in range(1, args.epochs + 1):
		epoch_start = time.time()
		train_m = train_one_epoch(
			model=model,
			loader=train_loader,
			device=device,
			criterion=criterion,
			optimizer=optimizer,
			scaler=scaler,
			use_amp=use_amp,
		)
		val_m = evaluate(model=model, loader=valid_loader, device=device, criterion=criterion, use_amp=use_amp)
		scheduler.step()

		current_lr = optimizer.param_groups[0]["lr"]
		dt = time.time() - epoch_start
		print(
			f"Epoch {epoch:03d}/{args.epochs} | lr={current_lr:.2e} | "
			f"train loss={train_m.loss:.4f} acc={train_m.acc*100:.2f}% | "
			f"val loss={val_m.loss:.4f} acc={val_m.acc*100:.2f}% | {dt:.1f}s"
		)

		improved = val_m.loss < best_val_loss - 1e-6
		if improved:
			best_val_loss = val_m.loss
			best_epoch = epoch
			save_best_checkpoint(epoch=epoch, val_loss=best_val_loss)
			print(f"  best updated -> val loss={best_val_loss:.4f} | saved: {out_path}")
			bad_epochs = 0
		else:
			bad_epochs += 1

		if bad_epochs >= int(args.patience):
			print(f"Early stopping at epoch {epoch} (best epoch {best_epoch}, best val loss {best_val_loss:.4f})")
			break

	# Load best checkpoint for final test evaluation
	if out_path.exists():
		ckpt = torch.load(str(out_path), map_location="cpu")
		model.load_state_dict(ckpt["model_state"])
		model.to(device)

	# Test
	test_m = evaluate(model=model, loader=test_loader, device=device, criterion=criterion, use_amp=use_amp)
	print(f"TEST | loss={test_m.loss:.4f} acc={test_m.acc*100:.2f}% (best val loss {best_val_loss:.4f} @ epoch {best_epoch})")

	print(f"Best checkpoint path: {out_path}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())

