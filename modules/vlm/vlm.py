import torch
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"

DEVICE = torch.device("mps" if torch.backends.mps.is_available()
                       else "cuda" if torch.cuda.is_available()
                       else "cpu")


def load_vlm():
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
    ).to(DEVICE)
    model.eval()
    return model, processor


def query(model, processor, image: Image.Image, prompt: str, max_new_tokens: int = 256) -> str:
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text",  "text":  prompt},
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, return_tensors="pt"
    ).to(DEVICE)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return processor.decode(
        output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip()


def describe_scene(model, processor, image: Image.Image, context: dict | None = None) -> str:
    prompt = "Describe what is happening in this tennis match image."
    if context:
        parts = []
        if "n_players" in context:
            parts.append(f"{context['n_players']} player(s) detected on court")
        if "scores" in context:
            scores = ", ".join(f"{s:.2f}" for s in context["scores"])
            parts.append(f"detection confidence: {scores}")
        if parts:
            prompt += f" Context: {'. '.join(parts)}."
    return query(model, processor, image, prompt)
