

"""
5.4 tests for the datasets construction
"""

from collections import Counter
from pathlib import Path
import pytest
from src.dataset_construction import (compute_cooccurrence, load_coco_subset, load_manifest)

ROOT = Path(__file__).resolve().parents[1]
ANNOTATIONS = ROOT / "instances_val2017.json"
IMAGE_DIR = ROOT / "val2017"
MANIFEST = ROOT / "data" / "manifest.csv"


@pytest.fixture(scope="module")
def coco():
    return load_coco_subset(str(ANNOTATIONS), str(IMAGE_DIR))

@pytest.fixture(scope="module")
def questions():
    return load_manifest(str(MANIFEST))

def test_question_types_are_balanced(questions):
    counts = Counter(question["question_type"] for question in questions)
    assert counts == {"present": 200, "absent_random": 200, "absent_adversarial": 200}

def test_no_image_category_pair_is_repeated(questions):
    pairs = [(question["image_id"], question["category"]) for question in questions]
    assert len(pairs) == len(set(pairs))

def test_present_categories_match_the_annotations(questions, coco):
    present_questions = [question for question in questions if question["question_type"] == "present"]

    for question in present_questions:
        present_categories = coco.image_to_categories[question["image_id"]]
        assert question["category"] in present_categories

def test_cooccurrence_is_deterministic(coco):
    first_result = compute_cooccurrence(coco)
    second_result = compute_cooccurrence(coco)

    assert first_result == second_result

def test_adversarial_categories_are_not_present(questions, coco):
    adversarial_questions = [question for question in questions if question["question_type"] == "absent_adversarial"]

    for question in adversarial_questions:
        present_categories = coco.image_to_categories[question["image_id"]]
        assert question["category"] not in present_categories
