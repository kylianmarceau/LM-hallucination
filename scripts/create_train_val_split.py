import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference import load_results
from src.probing import train_val_split

RESULTS_PATH = PROJECT_ROOT / "data" / "inference_results.pkl"
SEED_PATH = PROJECT_ROOT / "seed.txt"
SPLIT_PATH = PROJECT_ROOT / "data" / "train_val_split.json"
VAL_FRACTION = 0.30

def main():
    if SPLIT_PATH.exists():
        print(f"split already exists: {SPLIT_PATH}")
        return

    seed = int(SEED_PATH.read_text(encoding="utf-8").strip())
    results = load_results(str(RESULTS_PATH))
    train_indices, val_indices = train_val_split(results, VAL_FRACTION, seed)

    split = {
        "seed": seed,
        "val_fraction": VAL_FRACTION,
        "train_indices": train_indices,
        "val_indices": val_indices,
    }

    SPLIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    SPLIT_PATH.write_text(json.dumps(split, indent=2) + "\n", encoding="utf-8")

    train_counts = Counter(results[index].question_type for index in train_indices)
    val_counts = Counter(results[index].question_type for index in val_indices)

    print(f"training examples: {len(train_indices)}")
    print(f"validation examples: {len(val_indices)}")
    for question_type in sorted(train_counts):
        print(f"{question_type}: train={train_counts[question_type]}, val={val_counts[question_type]}")
    print(f"split saved to: {SPLIT_PATH}")

if __name__ == "__main__":
    main()