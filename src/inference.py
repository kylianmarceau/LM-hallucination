import pickle
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

class InferenceResult:
    def __init__(self, image_id, category, question_type, ground_truth, generated_text, parsed_answer, confidence, hidden_states):
        self.image_id = image_id
        self.category = category
        self.question_type = question_type
        self.ground_truth = ground_truth
        self.generated_text = generated_text
        self.parsed_answer = parsed_answer
        self.confidence = confidence
        self.hidden_states = hidden_states

def load_model(model_name, device):
    # use float32 on cpu and apple for compatibility
    dtype = torch.float16 if device.startswith("cuda") else torch.float32

    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForImageTextToText.from_pretrained(model_name, dtype=dtype, attn_implementation="eager")
    model.to(device)
    model.eval()

    return model, processor

def parse_answer(text):
    # only accept answers that clearly start with yes or no
    cleaned_text = text.strip().lower()
    cleaned_text = re.sub(r"^assistant\s*:\s*", "", cleaned_text)
    match = re.match(r"^(yes|no)\b", cleaned_text)

    if match is None:
        return None

    return match.group(1) == "yes"

def run_single_example(model, processor, image, question, vision_cache=None):
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=prompt, images=[image], return_tensors="pt").to(model.device)

    with torch.inference_mode():
        # reuse the image encoding for questions about the same image
        if vision_cache is not None:
            if "features" not in vision_cache:
                image_output = model.get_image_features(pixel_values=inputs["pixel_values"], pixel_attention_mask=inputs.get("pixel_attention_mask"))
                vision_cache["features"] = image_output

            inputs.pop("pixel_values")
            inputs.pop("pixel_attention_mask", None)
            inputs["image_hidden_states"] = vision_cache["features"]

        generation = model.generate(**inputs, max_new_tokens=6, do_sample=False, return_dict_in_generate=True, output_scores=True, output_hidden_states=True)

    prompt_length = inputs["input_ids"].shape[1]
    generated_ids = generation.sequences[0][prompt_length:]
    generated_text = processor.decode(generated_ids, skip_special_tokens=True).strip()

    # use the probability of the first generated answer token as confidence
    if len(generated_ids) == 0:
        confidence = 0.0
    else:
        first_token = generated_ids[0]
        confidence = torch.softmax(generation.scores[0][0].float(), dim=-1)[first_token].item()

    # skip the embedding output and keep each transformer layer unpooled
    layer_outputs = generation.hidden_states[0][1:]
    hidden_states = {}

    for layer_number, layer_output in enumerate(layer_outputs):
        hidden_states[layer_number] = layer_output[0].detach().cpu().numpy().astype(np.float16)

    return InferenceResult(-1, "", "", False, generated_text, parse_answer(generated_text), confidence, hidden_states)

def read_ground_truth(value):
    if isinstance(value, bool):
        return value

    return str(value).strip().lower() in {"true", "yes", "1"}

def run_inference_on_manifest(model, processor, manifest, image_dir):
    results = []
    current_image_id = None
    image = None
    vision_cache = None

    for record in manifest:
        image_id = int(record["image_id"])

        if image_id != current_image_id:
            image_path = Path(image_dir) / f"{image_id:012d}.jpg"

            if not image_path.is_file():
                raise FileNotFoundError(f"image not found: {image_path}")

            with Image.open(image_path) as opened_image:
                image = opened_image.convert("RGB")

            current_image_id = image_id
            vision_cache = {}

        result = run_single_example(model, processor, image, record["question"], vision_cache)
        result.image_id = image_id
        result.category = record["category"]
        result.question_type = record["question_type"]
        result.ground_truth = read_ground_truth(record["ground_truth"])
        results.append(result)

    return results

def save_results(results, path):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("wb") as output_file:
        pickle.dump(results, output_file)

def load_results(path):
    with Path(path).open("rb") as input_file:
        return pickle.load(input_file)