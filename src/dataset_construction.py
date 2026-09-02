
from __future__ import annotations
import csv
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
from pycocotools.coco import COCO

QUESTION_TEMPLATE = "Is there a {object} in this image? Answer yes or no."
QUESTION_TYPES = ("present", "absent_random", "absent_adversarial")
MANIFEST_FIELDS = ("image_id","category","question","question_type","ground_truth")

@dataclass(frozen=True)
class COCOSubset:
    coco: COCO
    annotation_path: Path
    image_dir: Path
    category_id_to_name: dict[int, str]
    category_name_to_id: dict[str, int]
    image_to_categories: dict[int, frozenset[str]]
    image_file_names: dict[int, str]

    @property
    def image_ids(self):
        """Return eligible image IDs in a stable order"""
        return tuple(sorted(self.image_to_categories))

    @property
    def categories(self):

        return tuple(sorted(self.category_name_to_id))

    def image_path(self, image_id: int):

        try:
            return self.image_dir / self.image_file_names[image_id]
        except KeyError as exc:
            raise KeyError(f"Unknown image ID: {image_id}") from exc


def load_coco_subset(
    annotation_path: str,
    image_dir: str,
) -> COCOSubset:
    
    annotation_file = Path(annotation_path).expanduser().resolve()
    images_directory = Path(image_dir).expanduser().resolve()

    if not annotation_file.is_file():
        raise FileNotFoundError(f"Annotation file not found: {annotation_file}")
    if not images_directory.is_dir():
        raise NotADirectoryError(f"Image directory not found: {images_directory}")

    coco = COCO(str(annotation_file))

    category_rows = coco.loadCats(coco.getCatIds())
    category_id_to_name = {int(category["id"]): str(category["name"]) for category in category_rows}
    category_name_to_id = {name: category_id for category_id, name in category_id_to_name.items()}
    if len(category_name_to_id) != len(category_id_to_name):
        raise ValueError("COCO category names must be unique")

    image_file_names = {int(image["id"]): str(image["file_name"]) for image in coco.dataset.get("images", [])}
    missing_images = [file_name for file_name in image_file_names.values() if not (images_directory / file_name).is_file()]
    if missing_images:
        preview = ", ".join(missing_images[:5])
        raise FileNotFoundError(f"{len(missing_images)} annotated images are missing from "f"{images_directory}. First missing files: {preview}")

    categories_by_image: dict[int, set[str]] = {image_id: set() for image_id in image_file_names}
    for annotation in coco.dataset.get("annotations", []):
        image_id = int(annotation["image_id"])
        category_id = int(annotation["category_id"])
        if image_id not in categories_by_image:
            raise ValueError(f"Annotation refers to unknown image ID {image_id}")
        try:
            category_name = category_id_to_name[category_id]
        except KeyError as exc:
            raise ValueError(
                f"Annotation refers to unknown category ID {category_id}"
            ) from exc
        categories_by_image[image_id].add(category_name)

    empty_image_ids = sorted(
        image_id
        for image_id, category_names in categories_by_image.items()
        if not category_names
    )
    if empty_image_ids:
        preview = ", ".join(map(str, empty_image_ids[:5]))
        raise ValueError(f"{len(empty_image_ids)} images have no instance categories and cannot " f"support a present question. First image IDs: {preview}")

    image_to_categories = {image_id: frozenset(category_names) for image_id, category_names in categories_by_image.items()}

    return COCOSubset(coco=coco,annotation_path=annotation_file,image_dir=images_directory,category_id_to_name=category_id_to_name,category_name_to_id=category_name_to_id,image_to_categories=image_to_categories,image_file_names=image_file_names)

def compute_cooccurrence(
    coco: COCOSubset,
) -> dict[tuple[str, str], int]:
        
    counts: defaultdict[tuple[str, str], int] = defaultdict(int)

    for image_id in coco.image_ids:
        present_categories = sorted(coco.image_to_categories[image_id])
        for category_a, category_b in combinations(present_categories, 2):
            counts[(category_a, category_b)] += 1
            counts[(category_b, category_a)] += 1

    return dict(counts)

def sample_image_ids(
    coco: COCOSubset,
    n_images: int,
    seed: int,
) -> list[int]:
    
    """Return a deterministic image-ID sample without replacement"""

    if n_images <= 0:
        raise ValueError("n_images must be positive")
    if seed < 0:
        raise ValueError("seed must be non-negative")

    eligible_image_ids = np.asarray(coco.image_ids, dtype=np.int64)
    if n_images > len(eligible_image_ids):
        raise ValueError(
            f"Cannot sample {n_images} images from a pool of "
            f"{len(eligible_image_ids)}"
        )

    rng = np.random.default_rng(seed=seed)
    sampled = rng.choice(eligible_image_ids, size=n_images, replace=False)
    return [int(image_id) for image_id in sampled]

def _cooccurrence_count(cooccurrence: dict[tuple[str, str], int],category_a: str,category_b: str,):
    """Read a count even if a caller supplied only one pair orientation."""

    return int(cooccurrence.get((category_a, category_b), cooccurrence.get((category_b, category_a), 0)))


def _choose_adversarial_category(present_categories: list[str],absent_categories: list[str],cooccurrence: dict[tuple[str, str], int],rng: np.random.Generator):

    candidate_details: dict[str, tuple[int, str]] = {}
    for absent_category in absent_categories:
        supporting_counts = [
            (
                _cooccurrence_count(
                    cooccurrence, absent_category, present_category
                ),
                present_category,
            )
            for present_category in present_categories
        ]
        best_count = max(count for count, _ in supporting_counts)
        supporting_present = min(
            present_category
            for count, present_category in supporting_counts
            if count == best_count
        )
        candidate_details[absent_category] = (best_count, supporting_present)

    maximum_count = max(count for count, _ in candidate_details.values())
    if maximum_count <= 0:
        raise ValueError(
            "No absent category has a positive co-occurrence count with any "
            "category present in the target image"
        )

    tied_candidates = sorted(
        category
        for category, (count, _) in candidate_details.items()
        if count == maximum_count
    )
    chosen_absent = str(rng.choice(tied_candidates))
    count, supporting_present = candidate_details[chosen_absent]
    return chosen_absent, supporting_present, count

def build_question_set(
    coco: COCOSubset,
    image_ids: list[int],
    cooccurrence: dict[tuple[str, str], int],
    seed: int,
) -> list[dict]:
    #return three deterministic existence-question records per image ID"""

    if seed < 0:
        raise ValueError("seed must be non-negative")
    if len(image_ids) != len(set(image_ids)):
        raise ValueError("image_ids must not contain duplicates")

    unknown_image_ids = sorted(set(image_ids) - set(coco.image_ids))
    if unknown_image_ids:
        preview = ", ".join(map(str, unknown_image_ids[:5]))
        raise ValueError(f"Unknown image IDs: {preview}")

    all_categories = set(coco.categories)
    rng = np.random.default_rng(seed=seed)
    questions: list[dict[str, Any]] = []

    for raw_image_id in image_ids:
        image_id = int(raw_image_id)
        present_categories = sorted(coco.image_to_categories[image_id])
        absent_categories = sorted(all_categories - set(present_categories))

        if not present_categories:
            raise ValueError(f"Image {image_id} has no present category")
        if len(absent_categories) < 2:
            raise ValueError(
                f"Image {image_id} needs at least two absent categories to "
                "construct distinct negative questions"
            )

        present_category = str(rng.choice(present_categories))
        (
            adversarial_category,
            _supporting_present_category,
            _cooccurrence_frequency,
        ) = _choose_adversarial_category(present_categories,absent_categories,cooccurrence,rng)

        random_candidates = [
            category
            for category in absent_categories
            if category != adversarial_category
        ]
        random_category = str(rng.choice(random_candidates))

        questions.extend(
            [
                {
                    "image_id": image_id,
                    "category": present_category,
                    "question": QUESTION_TEMPLATE.format(object=present_category),
                    "question_type": "present",
                    "ground_truth": True,
                },
                {
                    "image_id": image_id,
                    "category": random_category,
                    "question": QUESTION_TEMPLATE.format(object=random_category),
                    "question_type": "absent_random",
                    "ground_truth": False,
                },
                {
                    "image_id": image_id,
                    "category": adversarial_category,
                    "question": QUESTION_TEMPLATE.format(
                        object=adversarial_category
                    ),
                    "question_type": "absent_adversarial",
                    "ground_truth": False,
                },
            ]
        )

    return questions

def save_manifest(
    questions: list[dict],
    path: str,
) -> None:
    #Save question records as a byte reproducible CSV 

    output_path = Path(path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file,fieldnames=MANIFEST_FIELDS,extrasaction="ignore",lineterminator="\n")
        writer.writeheader()
        for question in questions:
            missing_fields = set(MANIFEST_FIELDS) - set(question)
            if missing_fields:
                missing = ", ".join(sorted(missing_fields))
                raise ValueError(f"Manifest record is missing fields: {missing}")
            writer.writerow(question)

def load_manifest(
    path: str,
) -> list[dict]:    
    """Load a CSV  and restore integer and Boolean field types"""

    input_path = Path(path).expanduser()
    questions: list[dict[str, Any]] = []

    with input_path.open("r", encoding="utf-8", newline="") as input_file:
        reader = csv.DictReader(input_file)
        if reader.fieldnames != list(MANIFEST_FIELDS):
            raise ValueError(
                f"Expected manifest columns {list(MANIFEST_FIELDS)}, "
                f"found {reader.fieldnames}"
            )

        for row_number, row in enumerate(reader, start=2):
            ground_truth_text = row["ground_truth"].strip().lower()
            if ground_truth_text in {"true", "1", "yes"}:
                ground_truth = True
            elif ground_truth_text in {"false", "0", "no"}:
                ground_truth = False
            else:
                raise ValueError(
                    f"Invalid ground_truth value on CSV row {row_number}: "
                    f"{row['ground_truth']!r}"
                )

            questions.append({"image_id": int(row["image_id"]),"category": row["category"],"question": row["question"],"question_type": row["question_type"],"ground_truth": ground_truth})

    return questions
