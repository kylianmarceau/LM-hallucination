import argparse
import sys
import time
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset_construction import load_manifest
from src.inference import load_model, run_inference_on_manifest, save_results

MODEL_NAME = "HuggingFaceTB/SmolVLM-256M-Instruct"
MANIFEST_PATH = ROOT / "data" / "manifest.csv"
IMAGE_DIR = ROOT / "val2017"

def choose_device(requested_device):
    if requested_device != "auto":
        return requested_device

    if torch.cuda.is_available():
        return "cuda"

    if torch.backends.mps.is_available():
        return "mps"

    return "cpu"

def select_debug_rows(manifest, image_count):
    image_ids = []

    for record in manifest:
        image_id = int(record["image_id"])

        if image_id not in image_ids:
            image_ids.append(image_id)

    selected_ids = set(image_ids[:image_count])
    return [record for record in manifest if int(record["image_id"]) in selected_ids]

def parse_args():
    parser = argparse.ArgumentParser(description="run smolvlm inference")
    parser.add_argument("--full", action="store_true", help="run all 600 questions")
    parser.add_argument("--debug-images", type=int, default=15, help="number of images used for a debug run")
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto", help="device used for inference")
    return parser.parse_args()

def main():
    args = parse_args()
    device = choose_device(args.device)
    manifest = load_manifest(str(MANIFEST_PATH))

    if args.full:
        selected_manifest = manifest
        output_path = ROOT / "data" / "inference_results.pkl"
    else:
        selected_manifest = select_debug_rows(manifest, args.debug_images)
        output_path = ROOT / "data" / "debug_inference_results.pkl"

    print(f"model: {MODEL_NAME}")
    print(f"device: {device}")
    print(f"questions: {len(selected_manifest)}")

    model, processor = load_model(MODEL_NAME, device)
    start_time = time.perf_counter()
    results = run_inference_on_manifest(model, processor, selected_manifest, str(IMAGE_DIR))
    runtime = time.perf_counter() - start_time
    save_results(results, str(output_path))

    first_result = results[0]
    first_layer = min(first_result.hidden_states)

    print(f"generated answer: {first_result.generated_text}")
    print(f"parsed answer: {first_result.parsed_answer}")
    print(f"confidence: {first_result.confidence:.4f}")
    print(f"hidden state shape: {first_result.hidden_states[first_layer].shape}")
    print(f"runtime: {runtime:.1f} seconds")
    print(f"saved results: {output_path}")

if __name__ == "__main__":
    main()
