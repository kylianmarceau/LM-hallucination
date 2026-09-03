from collections import defaultdict
import numpy as np

def train_val_split(results, val_fraction, seed):
    """Create one deterministic split, stratified by question type"""

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
