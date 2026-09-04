import csv
from pathlib import Path
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = PROJECT_ROOT / "data" / "layer_auroc.csv"
OUTPUT_PATH = PROJECT_ROOT / "figures" / "layer_performance.png"

def main():
    with RESULTS_PATH.open("r", encoding="utf-8", newline="") as input_file:
        rows = list(csv.DictReader(input_file))

    layers = [int(row["layer"]) for row in rows]
    accuracies = [float(row["accuracy"]) for row in rows]
    aurocs = [float(row["auroc"]) for row in rows]

    plt.figure(figsize=(8, 5))
    plt.plot(layers, accuracies, marker="o", label="Accuracy")
    plt.plot(layers, aurocs, marker="o", label="AUROC")
    plt.axhline(0.5, color="gray", linestyle="--", label="Random AUROC")
    plt.xlabel("Layer")
    plt.ylabel("Score")
    plt.title("Probe performance by layer")
    plt.ylim(0, 1)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PATH, dpi=300)
    print(f"plot saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
