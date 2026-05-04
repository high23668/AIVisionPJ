"""
Qwen3-VL Demo Script
- Visual grounding, VQA, scene understanding, 3D grounding
- Model: Qwen/Qwen3-VL-8B-Instruct (HuggingFace)
"""
import argparse
import time
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText


QWEN3VL_MODELS = {
    "2b": "Qwen/Qwen3-VL-2B-Instruct",
    "4b": "Qwen/Qwen3-VL-4B-Instruct",
    "8b": "Qwen/Qwen3-VL-8B-Instruct",
    "32b": "Qwen/Qwen3-VL-32B-Instruct",
}


def load_model(variant="4b", device="cuda", use_flash_attn=False, load_in_4bit=True):
    """Load Qwen3-VL model and processor.

    Args:
        variant: Model size - "2b", "4b", "8b", "32b"
        use_flash_attn: Use FlashAttention2 (requires flash-attn package)
        load_in_4bit: Use 4-bit NF4 quantization to fit 8B model in 16GB VRAM
                      8B bf16 = ~16GB, 8B 4bit = ~5GB
    """
    model_id = QWEN3VL_MODELS[variant]
    print(f"Loading Qwen3-VL-{variant.upper()}: {model_id}")
    size_map = {"2b": 5, "4b": 8, "8b": 16, "32b": 65}
    quant_note = " (4-bit quantized)" if load_in_4bit else ""
    print(f"  Model: ~{size_map.get(variant, 16)}GB bf16{quant_note}")

    processor = AutoProcessor.from_pretrained(model_id)

    # VRAM requirements by variant:
    #   2B  bf16=~5GB  4bit=~2GB  → fits on 16GB
    #   4B  bf16=~8GB  4bit=~3GB  → fits on 16GB (recommended for 16GB GPU)
    #   8B  bf16=~16GB 4bit=~5GB  → CAUTION: transformers 5.x + bitsandbytes 4bit
    #                                loading pipeline loads bf16 to VRAM before quantizing,
    #                                causing OOM on 16GB GPU. Needs 24GB+ or vLLM/llama.cpp.
    #   32B bf16=~65GB → requires multi-GPU or large VRAM

    load_kwargs = {
        "low_cpu_mem_usage": True,
    }

    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        load_kwargs["max_memory"] = {0: "12GiB", "cpu": "48GiB"}
        load_kwargs["device_map"] = "auto"
    else:
        load_kwargs["dtype"] = torch.bfloat16
        load_kwargs["device_map"] = "auto"

    if use_flash_attn:
        load_kwargs["attn_implementation"] = "flash_attention_2"

    model = AutoModelForImageTextToText.from_pretrained(model_id, **load_kwargs)
    model.eval()

    param_count = sum(p.numel() for p in model.parameters()) / 1e9
    vram_mb = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
    print(f"  OK - Loaded {param_count:.1f}B params | VRAM: {vram_mb:.0f}MB")
    return model, processor


def run_inference(model, processor, image, prompt, max_new_tokens=512, thinking=False, max_size=1024):
    """Run Qwen3-VL inference on an image with a text prompt."""
    if isinstance(image, (str, Path)):
        image = Image.open(image).convert("RGB")

    # 高解像度画像はVRAM不足になるので縮小
    if max(image.size) > max_size:
        image.thumbnail((max_size, max_size), Image.LANCZOS)
        print(f"  Image resized to {image.size}")

    # Build conversation
    enable_thinking = "/think" in prompt.lower() or thinking
    clean_prompt = prompt.replace("/think", "").strip()

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": clean_prompt},
            ],
        }
    ]

    # Apply chat template
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )

    inputs = processor(
        text=[text],
        images=[image],
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    t0 = time.perf_counter()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    elapsed_ms = (time.perf_counter() - t0) * 1000

    # Decode output (skip input tokens)
    generated = output_ids[:, inputs.input_ids.shape[1]:]
    response = processor.batch_decode(generated, skip_special_tokens=True)[0].strip()

    return response, elapsed_ms


def grounding_2d(model, processor, image, object_name, output_path=None):
    """2D visual grounding: find bounding box(es) and draw on image."""
    import re
    import subprocess

    if isinstance(image, (str, Path)):
        image = Image.open(image).convert("RGB")

    # 高解像度画像はVRAM不足になるので縮小
    if max(image.size) > 1024:
        image.thumbnail((1024, 1024), Image.LANCZOS)
        print(f"  Image resized to {image.size}")

    # Qwen3-VL native grounding: <ref>object</ref> 形式で座標を返す
    # coordinates are normalized to 0-1000
    W, H = image.size
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": (
                f"Locate all {object_name} in the image and output their bounding boxes. "
                f"Format each as JSON: {{\"bbox\": [x1, y1, x2, y2]}} where coordinates are "
                f"pixel values (image is {W}x{H}). Output only JSON array, nothing else."
            )},
        ],
    }]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inputs = processor(text=[text], images=[image], padding=True,
                       return_tensors="pt").to(model.device)

    t0 = time.perf_counter()
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=256, do_sample=False)
    ms = (time.perf_counter() - t0) * 1000

    # skip_special_tokens=False でbox tokenを保持
    generated = output_ids[:, inputs.input_ids.shape[1]:]
    raw = processor.batch_decode(generated, skip_special_tokens=False)[0].strip()
    print(f"  Raw response: {raw[:200]}")

    # JSON から bbox を抽出 (ピクセル座標)
    import json
    boxes = []
    # JSON配列を探す
    json_match = re.search(r'\[.*\]', raw, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group())
            for item in data:
                if "bbox" in item:
                    boxes.append(item["bbox"])
        except Exception:
            pass
    # フォールバック: [x1,y1,x2,y2] パターン
    if not boxes:
        raw_boxes = re.findall(r'\[(\d+)[,\s]+(\d+)[,\s]+(\d+)[,\s]+(\d+)\]', raw)
        boxes = [[int(v) for v in b] for b in raw_boxes]

    if not boxes:
        print(f"  BBoxが検出されませんでした。Raw: {raw[:200]}")
        return raw, ms

    draw_img = image.copy()
    from PIL import ImageDraw
    draw = ImageDraw.Draw(draw_img)

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = [int(v) for v in box]
        draw.rectangle([x1, y1, x2, y2], outline="lime", width=3)
        draw.text((x1 + 4, y1 + 4), f"{object_name} {i+1}", fill="lime")
        print(f"  Box {i+1}: ({x1},{y1})-({x2},{y2})")

    if output_path is None:
        stem = Path(object_name.replace(" ", "_"))
        output_path = Path("results") / f"qwen3vl_ground_{stem}.jpg"
    draw_img.save(str(output_path))
    print(f"  Saved: {output_path}")
    subprocess.Popen(["xdg-open", str(output_path)])

    return raw, ms


def scene_description(model, processor, image):
    """Generate scene description."""
    prompt = "Describe what you see in this image in detail, including all visible objects and their spatial relationships."
    response, ms = run_inference(model, processor, image, prompt, max_new_tokens=256)
    return response, ms


def robot_task_grounding(model, processor, image, instruction):
    """Interpret robot task instruction and identify target object."""
    prompt = f"""You are assisting a robot arm. Given the instruction: "{instruction}"
1. Identify the target object
2. Describe its location in the image
3. Estimate its approximate position (top/center/bottom, left/center/right)
Be concise."""
    response, ms = run_inference(model, processor, image, prompt, max_new_tokens=256)
    return response, ms


def vqa(model, processor, image, question):
    """Visual Question Answering."""
    response, ms = run_inference(model, processor, image, question, max_new_tokens=1024)
    return response, ms


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qwen3-VL Demo")
    parser.add_argument("--image", default="bus.jpg", help="Input image path")
    parser.add_argument("--variant", default="8b", choices=list(QWEN3VL_MODELS.keys()))
    parser.add_argument("--task", default="describe",
                        choices=["describe", "ground", "robot", "vqa", "benchmark"])
    parser.add_argument("--prompt", default="What objects are in this image?")
    parser.add_argument("--object", default="bus", help="Object to ground (for --task ground)")
    parser.add_argument("--max_tokens", type=int, default=512, help="最大出力トークン数 (デフォルト: 512)")
    args = parser.parse_args()

    model, processor = load_model(args.variant)
    image = Image.open(args.image).convert("RGB")

    if args.task == "describe":
        resp, ms = scene_description(model, processor, image)
        print(f"\nScene Description ({ms:.0f}ms):\n{resp}")

    elif args.task == "ground":
        resp, ms = grounding_2d(model, processor, image, args.object)
        print(f"\n2D Grounding ({ms:.0f}ms):\n{resp}")

    elif args.task == "robot":
        resp, ms = robot_task_grounding(model, processor, image, args.prompt)
        print(f"\nRobot Task Grounding ({ms:.0f}ms):\n{resp}")

    elif args.task == "vqa":
        resp, ms = vqa(model, processor, image, args.prompt)
        print(f"\nVQA Answer ({ms:.0f}ms):\n{resp}")

    elif args.task == "benchmark":
        print(f"\nBenchmarking Qwen3-VL-{args.variant.upper()}...")
        times = []
        for i in range(3):
            _, ms = vqa(model, processor, image, "How many objects are visible?")
            times.append(ms)
            print(f"  Run {i+1}: {ms:.0f}ms")
        avg_ms = np.mean(times)
        vram_mb = torch.cuda.memory_allocated() / 1024**2
        print(f"\nAvg: {avg_ms:.0f}ms | VRAM: {vram_mb:.0f}MB")
