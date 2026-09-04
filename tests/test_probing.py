from collections import Counter
from types import SimpleNamespace
from src.probing import train_val_split

def make_results(per_type=200):
    results = []
    for question_type in (
        "present",
        "absent_random",
        "absent_adversarial",
    ):
        for index in range(per_type):
            results.append(
                SimpleNamespace(
                    question_type=question_type,
                    parsed_answer=index % 10 != 0,
                    ground_truth=True,
                )
            )
    return results

def test_train_val_split_is_stratified_disjoint_and_complete():
    results = make_results()
    train_indices, val_indices = train_val_split(results, 0.30, seed=12345)

    assert len(train_indices) == 420
    assert len(val_indices) == 180
    assert set(train_indices).isdisjoint(val_indices)
    assert set(train_indices) | set(val_indices) == set(range(600))

    train_counts = Counter(results[index].question_type for index in train_indices)
    val_counts = Counter(results[index].question_type for index in val_indices)
    assert set(train_counts.values()) == {140}
    assert set(val_counts.values()) == {60}

def test_train_val_split_is_deterministic():
    results = make_results()
    first = train_val_split(results, 0.30, seed=1259156436113838583857)
    second = train_val_split(results, 0.30, seed=1259156436113838583857)
    assert first == second

def test_train_val_split_excludes_unclear_answers():
    results = make_results(per_type=10)
    results[3].parsed_answer = None

    train_indices, val_indices = train_val_split(results, 0.20, seed=7)

    assert 3 not in train_indices
    assert 3 not in val_indices
    assert set(train_indices).isdisjoint(val_indices)
    assert len(train_indices) + len(val_indices) == len(results) - 1
