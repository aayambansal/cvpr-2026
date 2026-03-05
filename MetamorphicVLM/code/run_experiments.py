"""
MetamorphicVLM: Metamorphic Testing for Vision-Language Model Robustness
=========================================================================
Full experiment pipeline deployed on Modal GPU.
Tests multiple VLMs against systematic image transformations.
"""

import modal
import json
import os
import io
import base64
import time

# ── Modal setup ──────────────────────────────────────────────────────────
app = modal.App("metamorphic-vlm")

vol = modal.Volume.from_name("metamorphic-vlm-data", create_if_missing=True)

base_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1-mesa-glx", "libglib2.0-0", "fonts-dejavu-core")
    .uv_pip_install(
        "Pillow>=10.0",
        "numpy>=1.24",
    )
)

model_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1-mesa-glx", "libglib2.0-0", "fonts-dejavu-core")
    .uv_pip_install(
        "torch==2.4.1",
        "torchvision==0.19.1",
        "transformers==4.46.3",
        "accelerate>=0.26.0",
        "Pillow>=10.0",
        "numpy>=1.24",
        "scipy>=1.11",
        "bitsandbytes>=0.43.0",
        "sentencepiece",
        "protobuf",
    )
)

hf_secret = modal.Secret.from_dict({"HF_TOKEN": os.environ.get("HF_TOKEN", "")})


# ── Image helpers ────────────────────────────────────────────────────────

def encode_image_base64(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# ── Prepare test data ────────────────────────────────────────────────────

@app.function(image=base_image, volumes={"/data": vol}, timeout=600)
def prepare_test_data():
    """Generate test images, apply transformations, save to volume as chunks."""
    from PIL import Image, ImageDraw, ImageFont, ImageFilter
    import numpy as np
    import math
    import json

    print("Generating test suite...")
    test_cases = []

    # ─── Category 1: Object Recognition (10 images) ───
    shapes_spec = [
        ("circle", "red"), ("square", "blue"), ("triangle", "green"),
        ("star", "yellow"), ("diamond", "purple"), ("rectangle", "orange"),
        ("circle", "blue"), ("square", "red"), ("triangle", "yellow"),
        ("diamond", "green"),
    ]
    color_map = {
        "red": (220, 50, 50), "blue": (50, 50, 220), "green": (50, 180, 50),
        "yellow": (220, 200, 50), "purple": (150, 50, 180), "orange": (240, 150, 30),
    }
    for i, (shape, color) in enumerate(shapes_spec):
        img = Image.new("RGB", (448, 448), "white")
        draw = ImageDraw.Draw(img)
        c = color_map[color]
        cx, cy = 224, 224
        if shape == "circle":
            draw.ellipse([cx-80, cy-80, cx+80, cy+80], fill=c)
        elif shape == "square":
            draw.rectangle([cx-80, cy-80, cx+80, cy+80], fill=c)
        elif shape == "triangle":
            draw.polygon([(cx, cy-90), (cx-90, cy+70), (cx+90, cy+70)], fill=c)
        elif shape == "star":
            pts = []
            for j in range(10):
                angle = math.pi/2 + j * math.pi / 5
                r = 90 if j % 2 == 0 else 40
                pts.append((cx + r*math.cos(angle), cy - r*math.sin(angle)))
            draw.polygon(pts, fill=c)
        elif shape == "diamond":
            draw.polygon([(cx, cy-90), (cx+70, cy), (cx, cy+90), (cx-70, cy)], fill=c)
        elif shape == "rectangle":
            draw.rectangle([cx-100, cy-50, cx+100, cy+50], fill=c)
        test_cases.append({
            "id": f"objrec_{i:02d}", "category": "object_recognition",
            "image": img,
            "question": "What shape is shown in the center of this image? Answer with just the shape name.",
            "answer": shape,
        })

    # ─── Category 2: Color Identification (10 images) ───
    colors_test = [
        ("red", (220, 50, 50)), ("blue", (50, 50, 220)), ("green", (50, 180, 50)),
        ("yellow", (220, 200, 50)), ("purple", (150, 50, 180)),
        ("orange", (240, 150, 30)), ("pink", (240, 130, 170)),
        ("brown", (140, 80, 30)), ("cyan", (50, 200, 200)),
        ("white", (240, 240, 240)),
    ]
    for i, (cname, cval) in enumerate(colors_test):
        img = Image.new("RGB", (448, 448), (80, 80, 80))
        draw = ImageDraw.Draw(img)
        draw.ellipse([124, 124, 324, 324], fill=cval)
        test_cases.append({
            "id": f"color_{i:02d}", "category": "color_identification",
            "image": img,
            "question": "What color is the circle in this image? Answer with just the color name.",
            "answer": cname,
        })

    # ─── Category 3: Counting (10 images) ───
    for i, count in enumerate([1, 2, 3, 4, 5, 6, 7, 8, 3, 5]):
        img = Image.new("RGB", (448, 448), "white")
        draw = ImageDraw.Draw(img)
        rng = np.random.RandomState(42 + i)
        positions = []
        for _ in range(count):
            for _ in range(200):
                x = rng.randint(60, 388)
                y = rng.randint(60, 388)
                if all(((x-px)**2 + (y-py)**2) > 3600 for px, py in positions):
                    positions.append((x, y))
                    break
        for px, py in positions:
            draw.ellipse([px-25, py-25, px+25, py+25], fill=(50, 50, 220))
        test_cases.append({
            "id": f"count_{i:02d}", "category": "counting",
            "image": img,
            "question": "How many blue circles are in this image? Answer with just the number.",
            "answer": str(count),
        })

    # ─── Category 4: Text Reading (10 images) ───
    words = ["HELLO", "WORLD", "SCIENCE", "ROBOT", "BRAIN",
             "VISION", "MODEL", "LEARN", "DATA", "TEST"]
    for i, word in enumerate(words):
        img = Image.new("RGB", (448, 448), "white")
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 60)
        except:
            font = ImageFont.load_default()
        bbox = draw.textbbox((0, 0), word, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text(((448 - tw) // 2, (448 - th) // 2), word, fill="black", font=font)
        test_cases.append({
            "id": f"text_{i:02d}", "category": "text_reading",
            "image": img,
            "question": "What word is written in this image? Answer with just the word.",
            "answer": word.lower(),
        })

    # ─── Category 5: Spatial Reasoning (10 images) ───
    spatial_configs = [
        ("above", (224, 120), (224, 320)), ("below", (224, 320), (224, 120)),
        ("left of", (120, 224), (328, 224)), ("right of", (328, 224), (120, 224)),
        ("above", (224, 100), (224, 340)), ("below", (224, 340), (224, 100)),
        ("left of", (100, 224), (348, 224)), ("right of", (348, 224), (100, 224)),
        ("above", (224, 130), (224, 310)), ("left of", (130, 224), (318, 224)),
    ]
    for i, (rel, red_pos, blue_pos) in enumerate(spatial_configs):
        img = Image.new("RGB", (448, 448), "white")
        draw = ImageDraw.Draw(img)
        rx, ry = red_pos
        bx, by = blue_pos
        draw.ellipse([rx-35, ry-35, rx+35, ry+35], fill=(220, 50, 50))
        draw.rectangle([bx-35, by-35, bx+35, by+35], fill=(50, 50, 220))
        test_cases.append({
            "id": f"spatial_{i:02d}", "category": "spatial_reasoning",
            "image": img,
            "question": "Is the red circle above, below, left of, or right of the blue square? Answer with just the spatial relation.",
            "answer": rel,
        })

    # ─── Category 6: Scene/Pattern (10 images) ───
    def make_grid(img, draw):
        for x in range(0, 448, 56):
            draw.line([(x, 0), (x, 447)], fill="black", width=2)
        for y in range(0, 448, 56):
            draw.line([(0, y), (447, y)], fill="black", width=2)

    def make_diag(img, draw):
        for x in range(-448, 448, 40):
            draw.line([(x, 0), (x+448, 448)], fill=(50, 50, 180), width=8)

    def make_circles(img, draw):
        for r in range(20, 220, 25):
            draw.ellipse([224-r, 224-r, 224+r, 224+r], outline=(r*2 % 255, 50, 200-r%200), width=3)

    def make_checker(img, draw):
        for x in range(0, 448, 56):
            for y in range(0, 448, 56):
                c = "black" if (x//56+y//56) % 2 == 0 else "white"
                draw.rectangle([x, y, x+56, y+56], fill=c)

    def make_radial(img, draw):
        for a_idx in range(24):
            a = a_idx * math.pi / 12
            draw.line([(224, 224), (int(224+200*math.cos(a)), int(224+200*math.sin(a)))], fill="black", width=2)

    def make_dots(img, draw):
        rng = np.random.RandomState(99)
        for x, y in zip(rng.randint(20, 428, 50), rng.randint(20, 428, 50)):
            draw.ellipse([x-4, y-4, x+4, y+4], fill=(0, 0, 0))

    def make_bars(img, draw):
        for y in range(0, 448, 45):
            draw.rectangle([0, y, 447, y+20], fill=(200, 50, 50))

    def make_nested(img, draw):
        for s in range(20, 210, 25):
            draw.rectangle([224-s, 224-s, 224+s, 224+s], outline=(s*2%255, 100, 50), width=3)

    def make_cross(img, draw):
        draw.rectangle([174, 0, 274, 447], fill=(50, 150, 50))
        draw.rectangle([0, 174, 447, 274], fill=(50, 150, 50))

    def make_spiral(img, draw):
        for t in range(10, 500, 8):
            x = int(224 + t * math.cos(t/5) * 0.4)
            y = int(224 + t * math.sin(t/5) * 0.4)
            draw.ellipse([x-5, y-5, x+5, y+5], fill="black")

    scenes = [
        ("grid", make_grid), ("diagonal stripes", make_diag),
        ("concentric circles", make_circles), ("checkerboard", make_checker),
        ("radial lines", make_radial), ("random dots", make_dots),
        ("horizontal bars", make_bars), ("nested squares", make_nested),
        ("cross", make_cross), ("spiral", make_spiral),
    ]
    for i, (desc, fn) in enumerate(scenes):
        img = Image.new("RGB", (448, 448), "white")
        draw = ImageDraw.Draw(img)
        fn(img, draw)
        test_cases.append({
            "id": f"scene_{i:02d}", "category": "scene_understanding",
            "image": img,
            "question": "Describe the main visual pattern in this image in 2-3 words.",
            "answer": desc,
        })

    print(f"Generated {len(test_cases)} base test cases")

    # ── Apply transformations ──
    def apply_transforms(img):
        W, H = img.size
        results = []
        # 1. Resize (downsample then upsample)
        for sev, scale in enumerate([0.75, 0.5, 0.35, 0.25], 1):
            small = img.resize((int(W*scale), int(H*scale)), Image.BILINEAR)
            restored = small.resize((W, H), Image.BILINEAR)
            results.append(("resize", sev, restored))
        # 2. Crop margins
        for sev, pct in enumerate([0.05, 0.10, 0.15, 0.20], 1):
            left, top = int(W*pct), int(H*pct)
            cropped = img.crop((left, top, W-left, H-top)).resize((W, H), Image.BILINEAR)
            results.append(("crop", sev, cropped))
        # 3. Rotation
        for sev, angle in enumerate([1, 2, 5, 10], 1):
            rotated = img.rotate(angle, resample=Image.BILINEAR, expand=False, fillcolor=(255,255,255))
            results.append(("rotation", sev, rotated))
        # 4. JPEG compression
        for sev, quality in enumerate([70, 50, 30, 10], 1):
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=quality)
            buf.seek(0)
            compressed = Image.open(buf).convert("RGB")
            results.append(("jpeg", sev, compressed))
        # 5. Gaussian blur
        for sev, radius in enumerate([0.5, 1.0, 2.0, 3.5], 1):
            blurred = img.filter(ImageFilter.GaussianBlur(radius=radius))
            results.append(("blur", sev, blurred))
        # 6. Border text
        texts = ["Page 1", "DRAFT - CONFIDENTIAL",
                 "Lorem ipsum dolor sit amet", "WARNING: TOP SECRET"]
        for sev, text in enumerate(texts, 1):
            pad = 15 * sev
            new_h = H + 2 * pad
            bordered = Image.new("RGB", (W, new_h), "white")
            bordered.paste(img, (0, pad))
            d = ImageDraw.Draw(bordered)
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10 + sev*2)
            except:
                font = ImageFont.load_default()
            d.text((5, 2), text, fill="gray", font=font)
            if sev >= 2:
                d.text((5, new_h - pad + 2), text, fill="gray", font=font)
            final = bordered.resize((W, H), Image.BILINEAR)
            results.append(("border_text", sev, final))
        return results

    # Build all items
    all_items = []
    for tc in test_cases:
        img = tc["image"]
        b64 = encode_image_base64(img)
        all_items.append({
            "id": tc["id"], "category": tc["category"],
            "image_b64": b64, "question": tc["question"],
            "gt_answer": tc["answer"],
            "transform_name": "original", "transform_severity": 0,
        })
        for tname, tsev, timg in apply_transforms(img):
            all_items.append({
                "id": tc["id"], "category": tc["category"],
                "image_b64": encode_image_base64(timg),
                "question": tc["question"], "gt_answer": tc["answer"],
                "transform_name": tname, "transform_severity": tsev,
            })

    print(f"Total items: {len(all_items)}")

    # Save in chunks (each ~50MB)
    chunk_size = 300
    os.makedirs("/data/chunks", exist_ok=True)
    n_chunks = 0
    for i in range(0, len(all_items), chunk_size):
        chunk = all_items[i:i+chunk_size]
        with open(f"/data/chunks/chunk_{n_chunks:03d}.json", "w") as f:
            json.dump(chunk, f)
        n_chunks += 1
    
    # Save metadata
    meta = {"n_items": len(all_items), "n_chunks": n_chunks, "chunk_size": chunk_size}
    with open("/data/metadata.json", "w") as f:
        json.dump(meta, f)
    vol.commit()
    print(f"Saved {n_chunks} chunks to volume")
    return meta


# ── Model Inference ──────────────────────────────────────────────────────

@app.function(
    gpu="A100",
    image=model_image,
    volumes={"/data": vol},
    timeout=10800,
    memory=65536,
    secrets=[hf_secret],
)
def run_model(model_key: str):
    """Run inference for one model across all test data chunks."""
    import torch
    import json
    import gc

    vol.reload()
    device = "cuda"
    
    # Load metadata
    with open("/data/metadata.json") as f:
        meta = json.load(f)
    
    print(f"[{model_key}] Loading model...")

    if model_key == "llava":
        from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
        model_id = "llava-hf/llava-v1.6-mistral-7b-hf"
        processor = LlavaNextProcessor.from_pretrained(model_id)
        model = LlavaNextForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.float16, device_map="auto"
        )
    elif model_key == "qwen2vl":
        from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
        model_id = "Qwen/Qwen2-VL-2B-Instruct"
        processor = Qwen2VLProcessor.from_pretrained(model_id)
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.float16, device_map="auto"
        )
    elif model_key == "phi3v":
        from transformers import AutoModelForCausalLM, AutoProcessor
        model_id = "microsoft/Phi-3.5-vision-instruct"
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float16, device_map="auto",
            trust_remote_code=True, _attn_implementation="eager"
        )
    else:
        raise ValueError(f"Unknown model: {model_key}")

    model.eval()
    print(f"[{model_key}] Model loaded. Processing chunks...")

    all_results = []

    for chunk_idx in range(meta["n_chunks"]):
        chunk_path = f"/data/chunks/chunk_{chunk_idx:03d}.json"
        with open(chunk_path) as f:
            chunk = json.load(f)

        for item in chunk:
            from PIL import Image as PILImage
            img_bytes = base64.b64decode(item["image_b64"])
            pil_img = PILImage.open(io.BytesIO(img_bytes)).convert("RGB")
            question = item["question"]

            try:
                if model_key == "llava":
                    conv = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
                    prompt = processor.apply_chat_template(conv, add_generation_prompt=True)
                    inputs = processor(images=pil_img, text=prompt, return_tensors="pt").to(device)
                    with torch.no_grad():
                        out = model.generate(**inputs, max_new_tokens=60, do_sample=False)
                    answer = processor.decode(out[0], skip_special_tokens=True)
                    if "[/INST]" in answer:
                        answer = answer.split("[/INST]")[-1].strip()

                elif model_key == "qwen2vl":
                    messages = [{"role": "user", "content": [
                        {"type": "image", "image": pil_img},
                        {"type": "text", "text": question},
                    ]}]
                    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    inputs = processor(text=[text], images=[pil_img], padding=True, return_tensors="pt").to(device)
                    with torch.no_grad():
                        gen_ids = model.generate(**inputs, max_new_tokens=60, do_sample=False)
                    trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, gen_ids)]
                    answer = processor.batch_decode(trimmed, skip_special_tokens=True)[0]

                elif model_key == "phi3v":
                    msgs = [{"role": "user", "content": f"<|image_1|>\n{question}"}]
                    prompt = processor.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                    inputs = processor(prompt, [pil_img], return_tensors="pt").to(device)
                    with torch.no_grad():
                        out = model.generate(**inputs, max_new_tokens=60, do_sample=False,
                                             eos_token_id=processor.tokenizer.eos_token_id)
                    answer = processor.decode(out[0], skip_special_tokens=True)
                    if "<|assistant|>" in answer:
                        answer = answer.split("<|assistant|>")[-1].strip()

                all_results.append({
                    "test_id": item["id"], "model": model_key,
                    "transform": item["transform_name"], "severity": item["transform_severity"],
                    "question": question, "gt_answer": item["gt_answer"],
                    "model_answer": answer.strip()[:200], "category": item["category"],
                    "success": True,
                })
            except Exception as e:
                all_results.append({
                    "test_id": item["id"], "model": model_key,
                    "transform": item["transform_name"], "severity": item["transform_severity"],
                    "question": question, "gt_answer": item["gt_answer"],
                    "model_answer": f"ERROR: {str(e)[:80]}", "category": item["category"],
                    "success": False,
                })

            # Clear GPU cache periodically
            if len(all_results) % 100 == 0:
                torch.cuda.empty_cache()

        print(f"  [{model_key}] Chunk {chunk_idx+1}/{meta['n_chunks']} done ({len(all_results)} results)")

    # Save results
    os.makedirs("/data/results", exist_ok=True)
    with open(f"/data/results/{model_key}.json", "w") as f:
        json.dump(all_results, f, indent=1)
    vol.commit()
    print(f"[{model_key}] Saved {len(all_results)} results")
    return {"model": model_key, "n_results": len(all_results)}


# ── Analysis ─────────────────────────────────────────────────────────────

@app.function(image=base_image, volumes={"/data": vol}, timeout=600)
def analyze_results():
    """Collect results from all models and compute robustness metrics."""
    import json
    import numpy as np

    vol.reload()
    all_results = []
    
    results_dir = "/data/results"
    for fname in os.listdir(results_dir):
        if fname.endswith(".json"):
            with open(f"{results_dir}/{fname}") as f:
                all_results.extend(json.load(f))

    print(f"Loaded {len(all_results)} total results")

    # ── Semantic matching ──
    def answers_match(gt, pred, category):
        gt_l = gt.lower().strip()
        pred_l = pred.lower().strip()
        if gt_l in pred_l:
            return True
        if category == "counting":
            num_words = {"one":"1","two":"2","three":"3","four":"4",
                         "five":"5","six":"6","seven":"7","eight":"8"}
            for w, n in num_words.items():
                if w in pred_l and gt_l == n:
                    return True
        if category == "spatial_reasoning":
            for kw in ["above", "below"]:
                if gt_l == kw and kw in pred_l:
                    return True
            if gt_l == "left of" and "left" in pred_l:
                return True
            if gt_l == "right of" and "right" in pred_l:
                return True
        if category == "text_reading":
            # case-insensitive exact
            if gt_l == pred_l.strip().strip('"').strip("'").lower():
                return True
        return False

    for r in all_results:
        r["correct"] = answers_match(r["gt_answer"], r["model_answer"], r["category"])

    models = sorted(set(r["model"] for r in all_results))
    transforms = ["original", "resize", "crop", "rotation", "jpeg", "blur", "border_text"]
    categories = sorted(set(r["category"] for r in all_results))

    # 1. Accuracy by model x transform x severity
    accuracy = {}
    for m in models:
        accuracy[m] = {}
        for t in transforms:
            accuracy[m][t] = {}
            sevs = [0] if t == "original" else [1,2,3,4]
            for s in sevs:
                sub = [r for r in all_results if r["model"]==m and r["transform"]==t and r["severity"]==s]
                if sub:
                    accuracy[m][t][str(s)] = round(sum(r["correct"] for r in sub)/len(sub), 4)

    # 2. Per-category accuracy drop
    cat_acc = {}
    for m in models:
        cat_acc[m] = {}
        for c in categories:
            orig = [r for r in all_results if r["model"]==m and r["category"]==c and r["transform"]=="original"]
            trans = [r for r in all_results if r["model"]==m and r["category"]==c and r["transform"]!="original"]
            o_acc = sum(r["correct"] for r in orig)/len(orig) if orig else 0
            t_acc = sum(r["correct"] for r in trans)/len(trans) if trans else 0
            cat_acc[m][c] = {"original": round(o_acc, 4), "transformed": round(t_acc, 4),
                             "drop": round(o_acc - t_acc, 4)}

    # 3. Consistency (same answer as original, regardless of correctness)
    consistency = {}
    for m in models:
        mr = [r for r in all_results if r["model"] == m]
        orig_answers = {r["test_id"]: r["model_answer"].lower().strip()[:80] for r in mr if r["transform"]=="original"}
        consistency[m] = {}
        for t in transforms:
            if t == "original": continue
            consistency[m][t] = {}
            for s in [1,2,3,4]:
                sub = [r for r in mr if r["transform"]==t and r["severity"]==s]
                total = 0; cons = 0
                for r in sub:
                    if r["test_id"] in orig_answers:
                        total += 1
                        ta = r["model_answer"].lower().strip()[:80]
                        oa = orig_answers[r["test_id"]]
                        if ta == oa or (len(oa) > 3 and oa[:25] == ta[:25]):
                            cons += 1
                if total:
                    consistency[m][t][str(s)] = round(cons/total, 4)

    # 4. MCI (Metamorphic Consistency Index) per model
    mci = {}
    for m in models:
        mr = [r for r in all_results if r["model"]==m]
        origs = {r["test_id"]: r["correct"] for r in mr if r["transform"]=="original"}
        total = 0; same = 0
        for r in mr:
            if r["transform"] == "original": continue
            if r["test_id"] in origs:
                total += 1
                if r["correct"] == origs[r["test_id"]]:
                    same += 1
        mci[m] = round(same/total, 4) if total else 0

    # 5. Breaking examples
    breaks = []
    for m in models:
        mr = [r for r in all_results if r["model"]==m]
        orig_correct_ids = {r["test_id"] for r in mr if r["transform"]=="original" and r["correct"]}
        orig_ans = {r["test_id"]: r["model_answer"] for r in mr if r["transform"]=="original"}
        for r in mr:
            if r["transform"] != "original" and r["test_id"] in orig_correct_ids and not r["correct"]:
                breaks.append({
                    "model": m, "test_id": r["test_id"], "category": r["category"],
                    "transform": r["transform"], "severity": r["severity"],
                    "gt": r["gt_answer"],
                    "orig_ans": orig_ans.get(r["test_id"], "")[:100],
                    "trans_ans": r["model_answer"][:100],
                })

    analysis = {
        "accuracy": accuracy, "cat_acc": cat_acc, "consistency": consistency,
        "mci": mci, "breaking_examples": breaks[:200],
        "models": models, "transforms": transforms, "categories": categories,
        "total_results": len(all_results),
    }
    with open("/data/analysis.json", "w") as f:
        json.dump(analysis, f, indent=2)
    vol.commit()
    print(f"Analysis saved. MCI: {mci}")
    return analysis


@app.local_entrypoint()
def main():
    print("=" * 60)
    print("MetamorphicVLM Experiment Pipeline")
    print("=" * 60)

    # Step 1: Generate test data
    print("\n[1/3] Generating test data & transformations...")
    meta = prepare_test_data.remote()
    print(f"  -> {meta['n_items']} items in {meta['n_chunks']} chunks")

    # Step 2: Run models (in parallel)
    models_to_test = ["llava", "qwen2vl", "phi3v"]
    print(f"\n[2/3] Running {len(models_to_test)} models...")
    handles = []
    for mk in models_to_test:
        print(f"  Spawning {mk}...")
        handles.append(run_model.spawn(mk))
    for h in handles:
        result = h.get()
        print(f"  -> {result['model']}: {result['n_results']} results")

    # Step 3: Analyze
    print("\n[3/3] Analyzing results...")
    analysis = analyze_results.remote()
    print(f"  MCI scores: {analysis['mci']}")
    print(f"  Breaking examples found: {len(analysis['breaking_examples'])}")
    print("\nDone!")
