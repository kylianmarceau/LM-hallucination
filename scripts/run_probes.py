import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference import load_results
from src.probing import build_feature_matrix, evaluate_probe, train_probe

RESULTS_PATH = PROJECT_ROOT / "data" / "inference_results.pkl"
SPLIT_PATH = PROJECT_ROOT / "data" / "train_val_split.json"
OUTPUT_PATH = PROJECT_ROOT / "data" / "layer_auroc.csv"
POOLING_STRATEGY = "last_token"

def main():
    results = load_results(str(RESULTS_PATH))
    split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))

    train_results = [results[index] for index in split["train_indices"]]
    val_results = [results[index] for index in split["val_indices"]]
    layers = sorted(results[0].hidden_states)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=["layer", "pooling_strategy", "accuracy", "auroc"])
        writer.writeheader()

        for layer in layers:
            X_train, y_train = build_feature_matrix(train_results, layer, POOLING_STRATEGY)
            X_val, y_val = build_feature_matrix(val_results, layer, POOLING_STRATEGY)

            probe = train_probe(X_train, y_train)
            metrics = evaluate_probe(probe, X_val, y_val)

            writer.writerow({"layer": layer,"pooling_strategy": POOLING_STRATEGY,"accuracy": metrics["accuracy"],"auroc": metrics["auroc"]})
            output_file.flush()

            print(f"layer {layer}: accuracy={metrics['accuracy']:.2%}, " f"auroc={metrics['auroc']:.4f}")

    print(f"results saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()  