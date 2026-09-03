import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.inference import load_results

RESULTS_PATH = ROOT / "data" / "inference_results.pkl"

def calculate_accuracy(results):
    clear_results = [result for result in results if result.parsed_answer is not None]
    correct_count = sum(result.parsed_answer == result.ground_truth for result in clear_results)

    if len(clear_results) == 0:
        return 0, 0, 0.0

    accuracy = correct_count / len(clear_results)
    return correct_count, len(clear_results), accuracy


def main():
    print("loading results...")
    results = load_results(str(RESULTS_PATH))

    unclear_count = sum(result.parsed_answer is None for result in results)
    unclear_rate = unclear_count / len(results)

    print(f"total results: {len(results)}")
    print(f"unclear answers: {unclear_count}")
    print(f"unclear rate: {unclear_rate:.2%}")

    correct, total, accuracy = calculate_accuracy(results)
    print(f"overall accuracy: {correct}/{total} ({accuracy:.2%})")

    question_types = ["present", "absent_random", "absent_adversarial"]

    for question_type in question_types:
        selected_results = [result for result in results if result.question_type == question_type]
        correct, total, accuracy = calculate_accuracy(selected_results)
        print(f"{question_type} accuracy: {correct}/{total} ({accuracy:.2%})")


if __name__ == "__main__":
    main()