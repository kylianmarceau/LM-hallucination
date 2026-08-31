"""Generate the seeded Part A question manifest"""

from __future__ import annotations
import argparse
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset_construction import (build_question_set,compute_cooccurrence,load_coco_subset,save_manifest,sample_image_ids)

def parse_args():
    """Parse command-line paths while keeping repository defaults convenient."""
    parser = argparse.ArgumentParser(description="Build the seeded 600-row COCO question manifest.")
    parser.add_argument("--annotation-path",type=Path,default=PROJECT_ROOT / "instances_val2017.json",help="Path to the COCO instances annotation JSON.")
    parser.add_argument("--image-dir",type=Path,default=PROJECT_ROOT / "val2017",help="Directory containing the COCO validation images.")
    
    parser.add_argument("--seed-file",type=Path,default=PROJECT_ROOT / "seed.txt",help="Text file containing the numeric seed from Section 5.2.")
    parser.add_argument("--output",type=Path,default=PROJECT_ROOT / "data" / "manifest.csv",help="Destination CSV manifest.")
    parser.add_argument("--n-images",type=int,default=200,help="Number of images to sample (the assignment requires 200).")
    return parser.parse_args()

def read_seed(path: Path):
    """Read and validate the numeric seed recorded in Section 5.2."""
 
    try:
        seed_text = path.read_text(encoding="utf-8").strip()
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Seed file not found: {path}") from exc

    if not seed_text:
        raise ValueError(f"Seed file is empty: {path}")

    try:
        seed = int(seed_text)
    except ValueError as exc:
        raise ValueError(
            f"Seed file must contain one integer, found {seed_text!r}"
        ) from exc

    if seed < 0:
        raise ValueError("Seed must be non-negative")
    return seed

def main():
    """Build and save the complete question manifest."""

    args = parse_args()
    seed = read_seed(args.seed_file)
    coco = load_coco_subset(annotation_path=str(args.annotation_path),image_dir=str(args.image_dir))
    cooccurrence = compute_cooccurrence(coco)
    image_ids = sample_image_ids(coco,n_images=args.n_images,seed=seed)
    questions = build_question_set(coco,image_ids=image_ids,cooccurrence=cooccurrence,seed=seed)
    save_manifest(questions, str(args.output))

    counts = Counter(question["question_type"] for question in questions)
    print(f"Seed: {seed}")
    print(f"Sampled images: {len(image_ids)}")
    print(f"Question records: {len(questions)}")
    for question_type in ("present", "absent_random", "absent_adversarial"):
        print(f"  {question_type}: {counts[question_type]}")
    print(f"Manifest saved to: {args.output.resolve()}")

if __name__ == "__main__":
    main()