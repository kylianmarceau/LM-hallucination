"""script for running the 8.1 experiments"""

import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.generalization import cross_category_split
from src.inference import load_results
from src.probing import build_feature_matrix, evaluate_probe, train_probe

RESULTS_PATH = PROJECT_ROOT / "data" / "inference_results.pkl"
SPLIT_PATH = PROJECT_ROOT / "data" / "train_val_split.json"
LAYER_RESULTS_PATH = PROJECT_ROOT / "data" / "layer_auroc.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "generalization_results.csv"
PREDICTIONS_PATH = PROJECT_ROOT / "data" / "generalization_predictions.csv"
LOG_PATH = PROJECT_ROOT / "logs" / "generalization_runs.jsonl"

def validate_saved_split(results, split):
    # check the saved indices before training either probe
    train_indices = split["train_indices"]
    val_indices = split["val_indices"]

    for indices in (train_indices, val_indices):
        if not indices:
            raise ValueError("the saved split contains an empty set")
        for index in indices:
            if type(index) is not int or not 0 <= index < len(results):
                raise ValueError("the saved split contains an invalid index")
        if len(indices) != len(set(indices)):
            raise ValueError("the saved split contains duplicate indices")

    train_set = set(train_indices)
    val_set = set(val_indices)
    if train_set & val_set:
        raise ValueError("the saved training and validation sets overlap")

    clear_indices = {index for index, result in enumerate(results) if result.parsed_answer is not None}
    if (train_set | val_set) != clear_indices:
        raise ValueError("the saved split does not match the clear cached results")

def save_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def main():
    # use the best layer from the saved probing results
    with LAYER_RESULTS_PATH.open("r", encoding="utf-8", newline="") as input_file:
        layer_rows = list(csv.DictReader(input_file))

    best_layer = max(layer_rows, key=lambda row: float(row["auroc"]))
    layer = int(best_layer["layer"])
    pooling_strategy = best_layer["pooling_strategy"]

    split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    print("loading cached inference results", flush=True)
    results = load_results(str(RESULTS_PATH))
    validate_saved_split(results, split)
    saved_train = set(split["train_indices"])
    saved_val = set(split["val_indices"])

    experiments = [("within_category", ["absent_adversarial"]), ("cross_category", ["present", "absent_random"])]
    test_types = ["absent_adversarial"]
    experiment_splits = []

    for name, train_types in experiments:
        train_indices, test_indices = cross_category_split(results, train_types, test_types, split["val_indices"])

        # every test example held out in part c
        if not set(train_indices) <= saved_train or not set(test_indices) <= saved_val:
            raise ValueError("an experiment does not follow the saved split")
        if set(test_indices) & saved_train:
            raise ValueError("test examples overlap with part c training examples")

        experiment_splits.append((name, train_types, train_indices, test_indices))

    if experiment_splits[0][3] != experiment_splits[1][3]:
        raise ValueError("both experiments must use the same adversarial test examples")

    test_indices = experiment_splits[0][3]
    test_results = [results[index] for index in test_indices]
    X_test, y_test = build_feature_matrix(test_results, layer, pooling_strategy)

    if len(np.unique(y_test)) != 2:
        raise ValueError("the test set needs both correctness classes for auroc")

    print(f"layer: {layer}, pooling: {pooling_strategy}")
    print(f"test examples: {len(y_test)}, hallucinated: {int((y_test == 0).sum())}")
    summary_rows = []
    prediction_rows = []

    for name, train_types, train_indices, _ in experiment_splits:
        started_at = datetime.now(timezone.utc).isoformat()
        start_time = time.perf_counter()
        train_results = [results[index] for index in train_indices]
        X_train, y_train = build_feature_matrix(train_results, layer, pooling_strategy)

        if len(np.unique(y_train)) != 2:
            raise ValueError(f"{name}: training needs both correctness classes")

        probe = train_probe(X_train, y_train)
        metrics = evaluate_probe(probe, X_test, y_test)
        predictions = probe.predict(X_test)
        probabilities = probe.predict_proba(X_test)[:, 1]
        runtime = time.perf_counter() - start_time

        row = {}
        row["experiment"] = name
        row["layer"] = layer
        row["pooling_strategy"] = pooling_strategy
        row["train_types"] = "+".join(train_types)
        row["test_types"] = "+".join(test_types)
        row["train_examples"] = len(y_train)
        row["test_examples"] = len(y_test)
        row["train_hallucinated"] = int((y_train == 0).sum())
        row["test_hallucinated"] = int((y_test == 0).sum())
        row["accuracy"] = float(metrics["accuracy"])
        row["auroc"] = float(metrics["auroc"])
        row["runtime_seconds"] = runtime
        summary_rows.append(row)

        # save the predictions with their original result indices
        for position, index in enumerate(test_indices):
            result = results[index]
            prediction_row = {}
            prediction_row["experiment"] = name
            prediction_row["result_index"] = index
            prediction_row["image_id"] = result.image_id
            prediction_row["category"] = result.category
            prediction_row["question_type"] = result.question_type
            prediction_row["ground_truth"] = result.ground_truth
            prediction_row["parsed_answer"] = result.parsed_answer
            prediction_row["confidence"] = result.confidence
            prediction_row["grounded_label"] = int(y_test[position])
            prediction_row["probe_prediction"] = int(predictions[position])
            prediction_row["grounded_probability"] = float(probabilities[position])
            prediction_rows.append(prediction_row)

        # add each timed probe run to the log
        log_entry = row.copy()
        log_entry["started_at_utc"] = started_at
        log_entry["seed"] = split["seed"]
        log_entry["split_file"] = str(SPLIT_PATH.relative_to(PROJECT_ROOT))
        log_entry["train_indices"] = train_indices
        log_entry["test_indices"] = test_indices
        log_entry["classifier"] = "LogisticRegression"
        log_entry["classifier_parameters"] = probe.get_params()
        log_entry["note"] = "same adversarial test set for both probes, 1 means grounded"
        log_entry["timing_note"] = "includes training feature construction, fitting and evaluation, excludes cache loading and test feature construction"
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

        with LOG_PATH.open("a", encoding="utf-8") as log_file:
            log_file.write(json.dumps(log_entry) + "\n")

        print(f"{name}: train={len(y_train)}, accuracy={metrics['accuracy']:.2%}, auroc={metrics['auroc']:.4f}, runtime={runtime:.2f}s")

    # positive gaps mean performance dropped when training on present and random questions

    within, cross = summary_rows
    for row in summary_rows:
        row["accuracy_gap_from_within"] = within["accuracy"] - row["accuracy"]
        row["auroc_gap_from_within"] = within["auroc"] - row["auroc"]

    save_csv(summary_rows, OUTPUT_PATH)
    save_csv(prediction_rows, PREDICTIONS_PATH)

    print(f"generalisation gaps: accuracy={cross['accuracy_gap_from_within']:.4f}, auroc={cross['auroc_gap_from_within']:.4f}")
    print(f"comparison saved to: {OUTPUT_PATH}")
    print(f"predictions saved to: {PREDICTIONS_PATH}")
    print(f"experiment log: {LOG_PATH}")

if __name__ == "__main__":
    main()
