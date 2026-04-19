"""Check for (near-)duplicate images across train/valid/test splits.

Why:
- For augmented datasets, random split by file can leak near-duplicates (same original image
  with different augmentation) across splits, inflating accuracy.

Method:
- Computes perceptual dHash for images and reports cross-split collisions.
- Optionally performs a lightweight near-duplicate search within buckets.

Usage:
  python near_duplicate_check.py --data_root "D:\\...\\New Plant Diseases Dataset(Augmented)\\New Plant Diseases Dataset(Augmented)"

Tips (Windows stability):
- This script is single-process by design.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp"}


def default_data_root() -> Path:
	here = Path(__file__).resolve().parent
	return (
		here
		/ "archive1"
		/ "New Plant Diseases Dataset(Augmented)"
		/ "New Plant Diseases Dataset(Augmented)"
	)


def iter_images(root: Path) -> list[Path]:
	paths: list[Path] = []
	for p in root.rglob("*"):
		if p.is_file() and p.suffix.lower() in IMG_EXTS:
			paths.append(p)
	return paths


def dhash_64(image_path: Path, hash_size: int = 8) -> int:
	"""Compute 64-bit dHash.

	Algorithm: resize to (hash_size+1, hash_size), convert to grayscale,
	compare adjacent pixels horizontally.
	"""
	try:
		from PIL import Image
	except Exception as e:
		raise RuntimeError("Pillow is required (usually installed with torchvision). Install: pip install pillow") from e

	with Image.open(image_path) as img:
		img = img.convert("L")
		img = img.resize((hash_size + 1, hash_size))
		pixels = list(img.getdata())

	# pixels length = (hash_size+1)*hash_size
	bits = 0
	bit_index = 0
	for row in range(hash_size):
		off = row * (hash_size + 1)
		for col in range(hash_size):
			left = pixels[off + col]
			right = pixels[off + col + 1]
			if left > right:
				bits |= 1 << bit_index
			bit_index += 1
	return bits


def hamming_distance64(a: int, b: int) -> int:
	return (a ^ b).bit_count()


@dataclass(frozen=True)
class Hit:
	split: str
	relpath: str
	hash: str


def main(argv: list[str] | None = None) -> int:
	parser = argparse.ArgumentParser(description="Detect (near-)duplicate images across dataset splits")
	parser.add_argument("--data_root", type=Path, default=default_data_root(), help="Folder containing train/valid/test")
	parser.add_argument("--sample_per_split", type=int, default=0, help="Randomly sample N images per split (0 = use all)")
	parser.add_argument("--seed", type=int, default=42, help="Random seed used for sampling")
	parser.add_argument("--bucket_prefix_bits", type=int, default=16, help="Near-dup search bucket prefix bits (0 disables near-dup search)")
	parser.add_argument("--near_hamming", type=int, default=0, help="Near-duplicate threshold by Hamming distance (0 disables near-dup search)")
	parser.add_argument("--output", type=Path, default=Path("near_duplicate_report.json"), help="Output JSON report path")
	args = parser.parse_args(argv)

	data_root = args.data_root.expanduser().resolve()
	splits = {"train": data_root / "train", "valid": data_root / "valid", "test": data_root / "test"}
	for name, p in splits.items():
		if not p.is_dir():
			raise FileNotFoundError(f"Missing split dir: {name} -> {p}")

	random.seed(int(args.seed))

	# Collect image paths
	split_paths: dict[str, list[Path]] = {}
	for split_name, split_dir in splits.items():
		paths = iter_images(split_dir)
		if int(args.sample_per_split) > 0 and len(paths) > int(args.sample_per_split):
			paths = random.sample(paths, int(args.sample_per_split))
		split_paths[split_name] = sorted(paths)
		print(f"{split_name}: {len(split_paths[split_name])} images")

	# Compute hashes
	hash_to_hits: dict[int, list[Hit]] = defaultdict(list)
	failures: list[str] = []
	for split_name, paths in split_paths.items():
		base = splits[split_name]
		for p in paths:
			try:
				h = dhash_64(p)
			except Exception as e:
				failures.append(f"{split_name}:{p} -> {type(e).__name__}: {e}")
				continue
			rel = str(p.relative_to(base)).replace("\\", "/")
			hash_to_hits[h].append(Hit(split=split_name, relpath=rel, hash=f"{h:016x}"))

	# Exact collisions across splits
	exact_collisions: list[dict] = []
	for h, hits in hash_to_hits.items():
		if len(hits) < 2:
			continue
		splits_in = sorted({x.split for x in hits})
		if len(splits_in) < 2:
			continue
		exact_collisions.append({
			"hash": f"{h:016x}",
			"splits": splits_in,
			"items": [x.__dict__ for x in hits],
		})

	exact_collisions.sort(key=lambda d: (len(d["splits"]), len(d["items"])), reverse=True)

	# Lightweight near-duplicate search (optional)
	near_pairs: list[dict] = []
	prefix_bits = int(args.bucket_prefix_bits)
	near_thr = int(args.near_hamming)
	if prefix_bits > 0 and near_thr > 0:
		mask = ((1 << prefix_bits) - 1) << (64 - prefix_bits)
		buckets: dict[int, list[tuple[int, Hit]]] = defaultdict(list)
		for h, hits in hash_to_hits.items():
			# keep one representative per hit
			for hit in hits:
				buckets[h & mask].append((h, hit))

		# Compare within each bucket; stop if too large
		max_bucket = 5000
		for b, items in buckets.items():
			if len(items) > max_bucket:
				continue
			for i in range(len(items)):
				h1, hit1 = items[i]
				for j in range(i + 1, len(items)):
					h2, hit2 = items[j]
					if hit1.split == hit2.split:
						continue
					d = hamming_distance64(h1, h2)
					if d <= near_thr:
						near_pairs.append({
							"hamming": int(d),
							"a": hit1.__dict__,
							"b": hit2.__dict__,
						})
		near_pairs.sort(key=lambda x: x["hamming"])

	report = {
		"data_root": str(data_root),
		"sample_per_split": int(args.sample_per_split),
		"exact_collision_groups": len(exact_collisions),
		"near_pairs": len(near_pairs),
		"failures": failures[:50],
		"exact_collisions": exact_collisions[:200],
		"near_pairs_preview": near_pairs[:200],
	}

	out_path = args.output if args.output.is_absolute() else (Path(__file__).resolve().parent / args.output)
	out_path = out_path.resolve()
	out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

	print("\nSummary")
	print(f"- exact collision groups (cross-split): {len(exact_collisions)}")
	if prefix_bits > 0 and near_thr > 0:
		print(f"- near-duplicate pairs (cross-split, hamming<={near_thr}): {len(near_pairs)}")
	if failures:
		print(f"- decode failures: {len(failures)} (showing up to 50 in JSON)")
	print(f"\nSaved: {out_path}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
