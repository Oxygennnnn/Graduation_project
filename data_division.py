"""Dataset split utility (ImageFolder style).

Goal
----
Convert a *flat* ImageFolder dataset:

  data_root/
    class_a/...
    class_b/...

into a split layout:

  data_root/
    train/class_a/...
    valid/class_a/...
    test/class_a/...

Default split ratios: train:valid:test = 7:2:1.

This script is designed for Windows-friendly usage with explicit safety checks
to avoid mixing with an existing non-empty split.
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
from pathlib import Path


IMAGE_EXTS = {
	".jpg",
	".jpeg",
	".png",
	".bmp",
	".tif",
	".tiff",
	".webp",
}


def _parse_ratios(text: str) -> tuple[int, int, int]:
	parts = [p.strip() for p in text.replace(":", ",").split(",") if p.strip()]
	if len(parts) != 3:
		raise ValueError("ratios must have 3 numbers, e.g. 7,2,1")
	try:
		r = tuple(int(x) for x in parts)
	except Exception as e:
		raise ValueError("ratios must be integers, e.g. 7,2,1") from e
	if any(x < 0 for x in r) or sum(r) <= 0:
		raise ValueError("ratios must be non-negative and not all zero")
	return r  # type: ignore[return-value]


def _dir_has_any_files(path: Path) -> bool:
	if not path.exists():
		return False
	for p in path.rglob("*"):
		if p.is_file():
			return True
	return False


def _list_images(folder: Path) -> list[Path]:
	imgs: list[Path] = []
	for p in folder.rglob("*"):
		if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
			imgs.append(p)
	return imgs


def _split_counts(total: int, ratios: tuple[int, int, int]) -> tuple[int, int, int]:
	if total <= 0:
		return (0, 0, 0)
	tr, va, te = ratios
	s = tr + va + te
	# Deterministic rounding: floor for train/valid, remainder to test.
	train_n = int(total * (tr / s)) if s > 0 else 0
	valid_n = int(total * (va / s)) if s > 0 else 0
	test_n = total - train_n - valid_n
	# Avoid negative on tiny totals.
	train_n = max(0, min(train_n, total))
	valid_n = max(0, min(valid_n, total - train_n))
	test_n = max(0, total - train_n - valid_n)
	return train_n, valid_n, test_n


def _ensure_unique_path(dst: Path) -> Path:
	"""If dst exists, append _1, _2, ... before suffix."""
	if not dst.exists():
		return dst
	stem = dst.stem
	suffix = dst.suffix
	parent = dst.parent
	for i in range(1, 10_000):
		cand = parent / f"{stem}_{i}{suffix}"
		if not cand.exists():
			return cand
	raise RuntimeError(f"Too many filename collisions under: {parent}")


def _move_or_copy(src: Path, dst: Path, copy: bool) -> None:
	dst.parent.mkdir(parents=True, exist_ok=True)
	dst = _ensure_unique_path(dst)
	if copy:
		shutil.copy2(str(src), str(dst))
	else:
		shutil.move(str(src), str(dst))


def split_dataset(
	data_root: Path,
	*,
	seed: int = 42,
	ratios: tuple[int, int, int] = (7, 2, 1),
	copy: bool = False,
	force: bool = False,
) -> None:
	"""Split dataset into train/valid/test under data_root.

	Supports two common layouts:
	A) data_root/train/<class>/... (will split from train into valid/test)
	B) data_root/<class>/...       (will create train/valid/test and split from flat)
	"""
	data_root = data_root.expanduser().resolve()
	if not data_root.is_dir():
		raise FileNotFoundError(f"data_root is not a directory: {data_root}")

	train_root = data_root / "train"
	valid_root = data_root / "valid"
	test_root = data_root / "test"

	# Safety: refuse to run if valid/test already contain files (unless force).
	if not force and (_dir_has_any_files(valid_root) or _dir_has_any_files(test_root)):
		raise RuntimeError(
			"valid/ or test/ is not empty. Use --force if you are sure you want to proceed. "
			f"valid={valid_root} test={test_root}"
		)

	rng = random.Random(int(seed))

	# Detect layout.
	if train_root.is_dir():
		# Layout A: split from existing train/<class>.
		class_dirs = [d for d in train_root.iterdir() if d.is_dir()]
		if not class_dirs:
			raise RuntimeError(f"No class folders under: {train_root}")
		source_class_root = train_root
	else:
		# Layout B: flat root (class folders directly under data_root).
		skip = {"train", "valid", "test"}
		class_dirs = [d for d in data_root.iterdir() if d.is_dir() and d.name not in skip]
		if not class_dirs:
			raise RuntimeError(f"No class folders found under: {data_root}")
		source_class_root = data_root
		train_root.mkdir(parents=True, exist_ok=True)

	valid_root.mkdir(parents=True, exist_ok=True)
	test_root.mkdir(parents=True, exist_ok=True)

	print(
		f"Splitting dataset under: {data_root}\n"
		f"  mode={'from train/' if source_class_root == train_root else 'from flat root'} | "
		f"ratios={ratios[0]}:{ratios[1]}:{ratios[2]} | seed={seed} | op={'copy' if copy else 'move'}"
	)

	for cls_dir in sorted(class_dirs, key=lambda p: p.name.lower()):
		cls_name = cls_dir.name
		images = _list_images(cls_dir)
		if not images:
			print(f"[skip] {cls_name}: no images")
			continue
		rng.shuffle(images)
		train_n, valid_n, test_n = _split_counts(len(images), ratios)

		dst_train = train_root / cls_name
		dst_valid = valid_root / cls_name
		dst_test = test_root / cls_name
		dst_train.mkdir(parents=True, exist_ok=True)
		dst_valid.mkdir(parents=True, exist_ok=True)
		dst_test.mkdir(parents=True, exist_ok=True)

		train_imgs = images[:train_n]
		valid_imgs = images[train_n : train_n + valid_n]
		test_imgs = images[train_n + valid_n :]

		for src in train_imgs:
			_move_or_copy(src, dst_train / src.name, copy=copy)
		for src in valid_imgs:
			_move_or_copy(src, dst_valid / src.name, copy=copy)
		for src in test_imgs:
			_move_or_copy(src, dst_test / src.name, copy=copy)

		# If we were splitting from flat root (Layout B), remove now-empty class dir.
		if source_class_root == data_root:
			try:
				# Only remove if fully empty.
				next(cls_dir.rglob("*"))
			except StopIteration:
				cls_dir.rmdir()
			else:
				# There are leftover files (non-images) - keep directory.
				pass

		print(
			f"[ok] {cls_name}: total={len(images)} -> train={len(train_imgs)} valid={len(valid_imgs)} test={len(test_imgs)}"
		)

	print("Done.")


def main(argv: list[str] | None = None) -> int:
	parser = argparse.ArgumentParser(description="Split ImageFolder dataset into train/valid/test (7:2:1)")
	parser.add_argument(
		"--data_root",
		type=Path,
		required=True,
		help="Dataset root. Either contains class folders, or contains train/<class>/...",
	)
	parser.add_argument("--seed", type=int, default=42, help="Random seed")
	parser.add_argument("--ratios", type=str, default="7,2,1", help="Split ratios like 7,2,1")
	parser.add_argument("--copy", action="store_true", help="Copy files instead of moving")
	parser.add_argument("--force", action="store_true", help="Proceed even if valid/test already have files")

	args = parser.parse_args(argv)
	ratios = _parse_ratios(str(args.ratios))
	split_dataset(
		Path(args.data_root),
		seed=int(args.seed),
		ratios=ratios,
		copy=bool(args.copy),
		force=bool(args.force),
	)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
