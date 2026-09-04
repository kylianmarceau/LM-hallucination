from collections import defaultdict
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

def pool_layer(hidden_states, strategy):
    """Pool a layer's token vectors into one feature vector."""

    if strategy == "last_token":
        return hidden_states[-1]

    raise ValueError(f"Unknown pooling strategy: {strategy}")

def build_feature_matrix(results, layer, strategy):
    """Build features and correctness labels for one layer."""

    features = []
    labels = []

    for result in results:
        if result.parsed_answer is None:
            continue

        features.append(pool_layer(result.hidden_states[layer], strategy))
        labels.append(int(result.parsed_answer == result.ground_truth))

    return np.asarray(features), np.asarray(labels)

def train_val_split(results, val_fraction, seed):
    """Create one deterministic split, stratified by question type."""

    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between 0 and 1")

    indices_by_type = defaultdict(list)
    for index, result in enumerate(results):
        if result.parsed_answer is not None:
            indices_by_type[result.question_type].append(index)

    if not indices_by_type:
        raise ValueError("No clear results are available for splitting")

    rng = np.random.default_rng(seed)
    train_indices = []
    val_indices = []

    for question_type in sorted(indices_by_type):
        type_indices = indices_by_type[question_type].copy()
        rng.shuffle(type_indices)

        val_count = round(len(type_indices) * val_fraction)
        if val_count <= 0 or val_count >= len(type_indices):
            raise ValueError(f"Not enough {question_type} results for this split")

        val_indices.extend(type_indices[:val_count])
        train_indices.extend(type_indices[val_count:])

    return sorted(train_indices), sorted(val_indices)

def train_probe(X_train, y_train):
    """Train a logistic-regression correctness probe."""

    probe = LogisticRegression(max_iter=1000, class_weight="balanced")
    probe.fit(X_train, y_train)
    return probe


def evaluate_probe(probe, X_val, y_val):
    """Return validation accuracy and AUROC."""

    predictions = probe.predict(X_val)
    probabilities = probe.predict_proba(X_val)[:, 1]

    return {
        "accuracy": accuracy_score(y_val, predictions),
        "auroc": roc_auc_score(y_val, probabilities),
    }