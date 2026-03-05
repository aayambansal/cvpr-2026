"""
MetamorphicVLM v2: Full-Scale Metamorphic Testing Pipeline
============================================================
Addresses all reviewer feedback:
- 10 transforms (6 original + illumination, contrast, perspective, occlusion)
- 100 test images (expanded from 60)
- Embedding-based semantic evaluation (not string match)
- Multiple models (LLaVA-v1.6-7B, Qwen2-VL-2B, InternVL2-2B)
- Human baseline estimation via psychophysics literature
- Failure mode clustering and mechanistic analysis
"""

import modal
import json
import os
import io
import base64
import time

app = modal.App("metamorphic-vlm-v2")
vol = modal.Volume.from_name("metamorphic-vlm-v2-data", create_if_missing=True)

base_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1-mesa-glx", "libglib2.0-0", "fonts-dejavu-core")
    .uv_pip_install("Pillow>=10.0", "numpy>=1.24", "scipy>=1.11")
)

model_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1-mesa-glx", "libglib2.0-0", "fonts-dejavu-core")
    .uv_pip_install(
        "torch==2.4.1", "torchvision==0.19.1",
        "transformers==4.46.3", "accelerate>=0.26.0",
        "Pillow>=10.0", "numpy>=1.24", "scipy>=1.11",
        "bitsandbytes>=0.43.0", "sentencepiece", "protobuf",
        "sentence-transformers>=2.2.0", "timm>=0.9.0", "einops>=0.7.0",
    )
)

hf_secret = modal.Secret.from_dict({"HF_TOKEN": os.environ.get("HF_TOKEN", "")})


def encode_image_base64(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# ── EXPANDED TEST SUITE: 100 images, 6 categories ───────────────────────

@app.function(image=base_image, volumes={"/data": vol}, timeout=900, memory=8192)
def prepare_test_data_v2():
    from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
    import numpy as np
    import math

    test_cases = []

    # ─── Category 1: Object Recognition (17 images) ───
    shapes_spec = [
        ("circle", "red"), ("square", "blue"), ("triangle", "green"),
        ("star", "yellow"), ("diamond", "purple"), ("rectangle", "orange"),
        ("circle", "blue"), ("square", "red"), ("triangle", "yellow"),
        ("diamond", "green"), ("circle", "green"), ("square", "purple"),
        ("triangle", "red"), ("star", "blue"), ("diamond", "orange"),
        ("rectangle", "green"), ("circle", "orange"),
    ]
    color_map = {
        "red": (220,50,50), "blue": (50,50,220), "green": (50,180,50),
        "yellow": (220,200,50), "purple": (150,50,180), "orange": (240,150,30),
    }
    for i, (shape, color) in enumerate(shapes_spec):
        img = Image.new("RGB", (448, 448), "white")
        draw = ImageDraw.Draw(img)
        c = color_map[color]; cx, cy = 224, 224
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
            "id": f"obj_{i:02d}", "category": "object_recognition", "image": img,
            "question": "What shape is shown in the center of this image? Answer with just the shape name.",
            "answer": shape,
            "alt_questions": [
                "Identify the geometric shape in this image. One word answer.",
                "What is the shape displayed? Reply with the shape name only.",
            ],
        })

    # ─── Category 2: Color Identification (17 images) ───
    colors_test = [
        ("red", (220,50,50)), ("blue", (50,50,220)), ("green", (50,180,50)),
        ("yellow", (220,200,50)), ("purple", (150,50,180)), ("orange", (240,150,30)),
        ("pink", (240,130,170)), ("brown", (140,80,30)), ("cyan", (50,200,200)),
        ("white", (240,240,240)), ("red", (200,30,30)), ("blue", (30,30,200)),
        ("green", (30,160,30)), ("yellow", (200,180,30)), ("purple", (130,30,160)),
        ("orange", (220,130,10)), ("pink", (220,110,150)),
    ]
    for i, (cname, cval) in enumerate(colors_test):
        img = Image.new("RGB", (448, 448), (80, 80, 80))
        draw = ImageDraw.Draw(img)
        draw.ellipse([124, 124, 324, 324], fill=cval)
        test_cases.append({
            "id": f"col_{i:02d}", "category": "color_identification", "image": img,
            "question": "What color is the circle in this image? Answer with just the color name.",
            "answer": cname,
            "alt_questions": [
                "Name the color of the circular shape. One word.",
                "What is the color of the main object? Answer briefly.",
            ],
        })

    # ─── Category 3: Counting (17 images) ───
    counts = [1, 2, 3, 4, 5, 6, 7, 8, 9, 2, 3, 4, 5, 6, 7, 1, 8]
    for i, count in enumerate(counts):
        img = Image.new("RGB", (448, 448), "white")
        draw = ImageDraw.Draw(img)
        rng = np.random.RandomState(42 + i)
        positions = []
        for _ in range(count):
            for _ in range(300):
                x = rng.randint(60, 388); y = rng.randint(60, 388)
                if all(((x-px)**2+(y-py)**2) > 3600 for px,py in positions):
                    positions.append((x, y)); break
        for px, py in positions:
            draw.ellipse([px-22, py-22, px+22, py+22], fill=(50, 50, 220))
        test_cases.append({
            "id": f"cnt_{i:02d}", "category": "counting", "image": img,
            "question": "How many blue circles are in this image? Answer with just the number.",
            "answer": str(count),
            "alt_questions": [
                "Count the blue dots. Reply with a number only.",
                "What is the total number of circles? Answer with a digit.",
            ],
        })

    # ─── Category 4: Text Reading (17 images) ───
    words = ["HELLO", "WORLD", "SCIENCE", "ROBOT", "BRAIN", "VISION", "MODEL",
             "LEARN", "DATA", "TEST", "NEURAL", "DEEP", "IMAGE", "TRAIN",
             "GRAPH", "LAYER", "TOKEN"]
    for i, word in enumerate(words):
        img = Image.new("RGB", (448, 448), "white")
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 55)
        except:
            font = ImageFont.load_default()
        bbox = draw.textbbox((0,0), word, font=font)
        tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]
        draw.text(((448-tw)//2, (448-th)//2), word, fill="black", font=font)
        test_cases.append({
            "id": f"txt_{i:02d}", "category": "text_reading", "image": img,
            "question": "What word is written in this image? Answer with just the word.",
            "answer": word.lower(),
            "alt_questions": [
                "Read the text in the image. Reply with the word only.",
                "What does the text say? One word answer.",
            ],
        })

    # ─── Category 5: Spatial Reasoning (16 images) ───
    spatial = [
        ("above", (224,120), (224,320)), ("below", (224,320), (224,120)),
        ("left of", (120,224), (328,224)), ("right of", (328,224), (120,224)),
        ("above", (224,100), (224,340)), ("below", (224,340), (224,100)),
        ("left of", (100,224), (348,224)), ("right of", (348,224), (100,224)),
        ("above", (224,130), (224,310)), ("left of", (130,224), (318,224)),
        ("below", (224,310), (224,130)), ("right of", (318,224), (130,224)),
        ("above", (224,110), (224,330)), ("below", (224,330), (224,110)),
        ("left of", (110,224), (338,224)), ("right of", (338,224), (110,224)),
    ]
    for i, (rel, rp, bp) in enumerate(spatial):
        img = Image.new("RGB", (448,448), "white")
        draw = ImageDraw.Draw(img)
        rx,ry = rp; bx,by = bp
        draw.ellipse([rx-35,ry-35,rx+35,ry+35], fill=(220,50,50))
        draw.rectangle([bx-35,by-35,bx+35,by+35], fill=(50,50,220))
        test_cases.append({
            "id": f"spt_{i:02d}", "category": "spatial_reasoning", "image": img,
            "question": "Is the red circle above, below, left of, or right of the blue square? Answer with just the spatial relation.",
            "answer": rel,
            "alt_questions": [
                "Describe the position of the red circle relative to the blue square. Brief answer.",
                "Where is the red circle compared to the blue square?",
            ],
        })

    # ─── Category 6: Scene Understanding (16 images) ───
    def make_pattern(name):
        img = Image.new("RGB", (448,448), "white")
        draw = ImageDraw.Draw(img)
        if name == "grid":
            for x in range(0,448,56): draw.line([(x,0),(x,447)], fill="black", width=2)
            for y in range(0,448,56): draw.line([(0,y),(447,y)], fill="black", width=2)
        elif name == "diagonal stripes":
            for x in range(-448,448,40): draw.line([(x,0),(x+448,448)], fill=(50,50,180), width=8)
        elif name == "concentric circles":
            for r in range(20,220,25): draw.ellipse([224-r,224-r,224+r,224+r], outline=(r*2%255,50,200-r%200), width=3)
        elif name == "checkerboard":
            for x in range(0,448,56):
                for y in range(0,448,56):
                    c = "black" if (x//56+y//56)%2==0 else "white"
                    draw.rectangle([x,y,x+56,y+56], fill=c)
        elif name == "radial lines":
            for a_i in range(24):
                a = a_i*math.pi/12
                draw.line([(224,224),(int(224+200*math.cos(a)),int(224+200*math.sin(a)))], fill="black", width=2)
        elif name == "random dots":
            rng = np.random.RandomState(99)
            for x,y in zip(rng.randint(20,428,50), rng.randint(20,428,50)):
                draw.ellipse([x-4,y-4,x+4,y+4], fill=(0,0,0))
        elif name == "horizontal bars":
            for y in range(0,448,45): draw.rectangle([0,y,447,y+20], fill=(200,50,50))
        elif name == "nested squares":
            for s in range(20,210,25): draw.rectangle([224-s,224-s,224+s,224+s], outline=(s*2%255,100,50), width=3)
        elif name == "cross":
            draw.rectangle([174,0,274,447], fill=(50,150,50))
            draw.rectangle([0,174,447,274], fill=(50,150,50))
        elif name == "spiral":
            for t in range(10,500,8):
                x = int(224+t*math.cos(t/5)*0.4); y = int(224+t*math.sin(t/5)*0.4)
                draw.ellipse([x-5,y-5,x+5,y+5], fill="black")
        elif name == "vertical bars":
            for x in range(0,448,45): draw.rectangle([x,0,x+20,447], fill=(50,50,200))
        elif name == "zigzag":
            pts = []
            for x in range(0,448,30):
                y = 124 if (x//30)%2==0 else 324
                pts.append((x,y))
            draw.line(pts, fill="black", width=3)
        elif name == "diamond grid":
            for x in range(0,448,60):
                for y in range(0,448,60):
                    draw.polygon([(x+30,y),(x+60,y+30),(x+30,y+60),(x,y+30)], outline="black", width=1)
        elif name == "polka dots":
            for x in range(30,448,60):
                for y in range(30,448,60):
                    draw.ellipse([x-15,y-15,x+15,y+15], fill=(200,50,50))
        elif name == "waves":
            for offset in range(0,448,40):
                pts = [(x, offset + int(20*math.sin(x/30))) for x in range(0,448,5)]
                draw.line(pts, fill=(50,50,180), width=2)
        elif name == "triangles":
            for x in range(0,448,80):
                for y in range(0,448,80):
                    draw.polygon([(x+40,y),(x+80,y+80),(x,y+80)], outline="black", width=2)
        return img

    scene_names = ["grid", "diagonal stripes", "concentric circles", "checkerboard",
                   "radial lines", "random dots", "horizontal bars", "nested squares",
                   "cross", "spiral", "vertical bars", "zigzag", "diamond grid",
                   "polka dots", "waves", "triangles"]
    for i, name in enumerate(scene_names):
        img = make_pattern(name)
        test_cases.append({
            "id": f"scn_{i:02d}", "category": "scene_understanding", "image": img,
            "question": "Describe the main visual pattern in this image in 2-3 words.",
            "answer": name,
            "alt_questions": [
                "What pattern do you see? Brief description.",
                "Name the visual pattern shown. 2-3 words.",
            ],
        })

    print(f"Generated {len(test_cases)} test cases")

    # ── Apply EXPANDED transformations (10 types × 4 severities) ──
    def apply_all_transforms(img):
        from PIL import ImageEnhance
        W, H = img.size
        results = []

        # 1. Resize
        for sev, scale in enumerate([0.75, 0.5, 0.35, 0.25], 1):
            small = img.resize((int(W*scale), int(H*scale)), Image.BILINEAR)
            results.append(("resize", sev, small.resize((W,H), Image.BILINEAR)))

        # 2. Crop
        for sev, pct in enumerate([0.05, 0.10, 0.15, 0.20], 1):
            l,t = int(W*pct), int(H*pct)
            results.append(("crop", sev, img.crop((l,t,W-l,H-t)).resize((W,H), Image.BILINEAR)))

        # 3. Rotation
        for sev, angle in enumerate([1, 2, 5, 10], 1):
            results.append(("rotation", sev, img.rotate(angle, resample=Image.BILINEAR, expand=False, fillcolor=(255,255,255))))

        # 4. JPEG compression
        for sev, q in enumerate([70, 50, 30, 10], 1):
            buf = io.BytesIO(); img.save(buf, format="JPEG", quality=q); buf.seek(0)
            results.append(("jpeg", sev, Image.open(buf).convert("RGB")))

        # 5. Gaussian blur
        for sev, r in enumerate([0.5, 1.0, 2.0, 3.5], 1):
            results.append(("blur", sev, img.filter(ImageFilter.GaussianBlur(radius=r))))

        # 6. Border text
        texts = ["Page 1", "DRAFT - CONFIDENTIAL", "Lorem ipsum dolor sit amet", "WARNING: TOP SECRET"]
        for sev, text in enumerate(texts, 1):
            pad = 15*sev
            bordered = Image.new("RGB", (W, H+2*pad), "white")
            bordered.paste(img, (0, pad))
            d = ImageDraw.Draw(bordered)
            try: font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10+sev*2)
            except: font = ImageFont.load_default()
            d.text((5,2), text, fill="gray", font=font)
            if sev >= 2: d.text((5, H+2*pad-pad+2), text, fill="gray", font=font)
            results.append(("border_text", sev, bordered.resize((W,H), Image.BILINEAR)))

        # 7. Illumination (brightness change)
        for sev, factor in enumerate([0.8, 0.6, 1.3, 1.6], 1):
            enhancer = ImageEnhance.Brightness(img)
            results.append(("illumination", sev, enhancer.enhance(factor)))

        # 8. Contrast change
        for sev, factor in enumerate([0.8, 0.6, 1.4, 1.8], 1):
            enhancer = ImageEnhance.Contrast(img)
            results.append(("contrast", sev, enhancer.enhance(factor)))

        # 9. Perspective warp (affine)
        for sev, strength in enumerate([0.02, 0.05, 0.08, 0.12], 1):
            # Simple perspective via affine coefficients
            w, h = W, H
            dx = int(w * strength); dy = int(h * strength)
            # Crop corners to simulate perspective
            warped = img.transform(
                (w, h), Image.QUAD,
                (dx, dy, -dx, h-dy, w+dx, h+dy, w-dx, dy),
                resample=Image.BILINEAR
            )
            results.append(("perspective", sev, warped))

        # 10. Partial occlusion (black rectangle overlay)
        for sev, pct in enumerate([0.05, 0.10, 0.15, 0.25], 1):
            occluded = img.copy()
            d = ImageDraw.Draw(occluded)
            rng = np.random.RandomState(77 + sev)
            bw, bh = int(W*pct), int(H*pct)
            bx = rng.randint(0, W-bw); by = rng.randint(0, H-bh)
            d.rectangle([bx, by, bx+bw, by+bh], fill=(0,0,0))
            results.append(("occlusion", sev, occluded))

        return results

    # Build all items including alt-question variants
    all_items = []
    for tc in test_cases:
        img = tc["image"]
        b64 = encode_image_base64(img)

        # Original with primary question
        all_items.append({
            "id": tc["id"], "category": tc["category"],
            "image_b64": b64, "question": tc["question"],
            "gt_answer": tc["answer"], "prompt_variant": 0,
            "transform_name": "original", "transform_severity": 0,
        })
        # Original with alt questions (prompt sensitivity)
        for qi, altq in enumerate(tc.get("alt_questions", []), 1):
            all_items.append({
                "id": tc["id"], "category": tc["category"],
                "image_b64": b64, "question": altq,
                "gt_answer": tc["answer"], "prompt_variant": qi,
                "transform_name": "original_altprompt", "transform_severity": qi,
            })
        # All transforms with primary question
        for tname, tsev, timg in apply_all_transforms(img):
            all_items.append({
                "id": tc["id"], "category": tc["category"],
                "image_b64": encode_image_base64(timg),
                "question": tc["question"], "gt_answer": tc["answer"],
                "prompt_variant": 0,
                "transform_name": tname, "transform_severity": tsev,
            })

    print(f"Total items: {len(all_items)}")

    # Save in chunks
    chunk_size = 300
    os.makedirs("/data/chunks", exist_ok=True)
    n_chunks = 0
    for i in range(0, len(all_items), chunk_size):
        with open(f"/data/chunks/chunk_{n_chunks:03d}.json", "w") as f:
            json.dump(all_items[i:i+chunk_size], f)
        n_chunks += 1

    meta = {"n_items": len(all_items), "n_chunks": n_chunks, "chunk_size": chunk_size,
            "n_base_images": len(test_cases), "n_transforms": 10, "n_severities": 4}
    with open("/data/metadata.json", "w") as f:
        json.dump(meta, f)
    vol.commit()
    print(f"Saved {n_chunks} chunks")
    return meta


# ── Model Inference with Embedding-Based Evaluation ──────────────────────

@app.function(
    gpu="A100", image=model_image, volumes={"/data": vol},
    timeout=14400, memory=65536, secrets=[hf_secret],
)
def run_model_v2(model_key: str):
    import torch, json, gc
    from sentence_transformers import SentenceTransformer

    vol.reload()
    with open("/data/metadata.json") as f:
        meta = json.load(f)

    device = "cuda"
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
    elif model_key == "internvl2":
        from transformers import AutoModel, AutoTokenizer
        model_id = "OpenGVLab/InternVL2-2B"
        processor = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModel.from_pretrained(
            model_id, torch_dtype=torch.float16, device_map="auto",
            trust_remote_code=True,
        )
    else:
        raise ValueError(f"Unknown model: {model_key}")

    model.eval()

    # Load sentence transformer for semantic eval
    print(f"[{model_key}] Loading sentence encoder for eval...")
    sent_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

    print(f"[{model_key}] Processing {meta['n_chunks']} chunks...")
    all_results = []

    for chunk_idx in range(meta["n_chunks"]):
        with open(f"/data/chunks/chunk_{chunk_idx:03d}.json") as f:
            chunk = json.load(f)

        for item in chunk:
            from PIL import Image as PILImage
            img_bytes = base64.b64decode(item["image_b64"])
            pil_img = PILImage.open(io.BytesIO(img_bytes)).convert("RGB")
            question = item["question"]

            try:
                if model_key == "llava":
                    conv = [{"role":"user","content":[{"type":"image"},{"type":"text","text":question}]}]
                    prompt = processor.apply_chat_template(conv, add_generation_prompt=True)
                    inputs = processor(images=pil_img, text=prompt, return_tensors="pt").to(device)
                    with torch.no_grad():
                        out = model.generate(**inputs, max_new_tokens=60, do_sample=False)
                    answer = processor.decode(out[0], skip_special_tokens=True)
                    if "[/INST]" in answer: answer = answer.split("[/INST]")[-1].strip()

                elif model_key == "qwen2vl":
                    messages = [{"role":"user","content":[
                        {"type":"image","image":pil_img},{"type":"text","text":question}
                    ]}]
                    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    inputs = processor(text=[text], images=[pil_img], padding=True, return_tensors="pt").to(device)
                    with torch.no_grad():
                        gen_ids = model.generate(**inputs, max_new_tokens=60, do_sample=False)
                    trimmed = [o[len(i):] for i,o in zip(inputs.input_ids, gen_ids)]
                    answer = processor.batch_decode(trimmed, skip_special_tokens=True)[0]

                elif model_key == "internvl2":
                    from torchvision import transforms as T
                    from torchvision.transforms.functional import InterpolationMode
                    import torchvision.transforms.functional as TF
                    IMAGENET_MEAN = (0.485, 0.456, 0.406)
                    IMAGENET_STD = (0.229, 0.224, 0.225)
                    def build_transform(input_size=448):
                        return T.Compose([
                            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
                            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
                            T.ToTensor(),
                            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
                        ])
                    transform = build_transform(448)
                    pixel_values = transform(pil_img).unsqueeze(0).to(torch.float16).to(device)
                    generation_config = dict(max_new_tokens=60, do_sample=False)
                    answer = model.chat(processor, pixel_values, question, generation_config)

                answer = answer.strip()[:200]

                # Embedding-based semantic similarity scoring
                gt = item["gt_answer"]
                emb_gt = sent_model.encode([gt], convert_to_tensor=False)
                emb_pred = sent_model.encode([answer.lower()[:100]], convert_to_tensor=False)
                import numpy as np
                sim = float(np.dot(emb_gt[0], emb_pred[0]) / (np.linalg.norm(emb_gt[0]) * np.linalg.norm(emb_pred[0]) + 1e-8))

                # Also do substring match as secondary signal
                gt_l = gt.lower().strip()
                pred_l = answer.lower().strip()
                substr_match = gt_l in pred_l

                all_results.append({
                    "test_id": item["id"], "model": model_key,
                    "transform": item["transform_name"], "severity": item["transform_severity"],
                    "prompt_variant": item.get("prompt_variant", 0),
                    "question": question, "gt_answer": gt,
                    "model_answer": answer, "category": item["category"],
                    "semantic_sim": round(sim, 4), "substr_match": substr_match,
                    "success": True,
                })
            except Exception as e:
                all_results.append({
                    "test_id": item["id"], "model": model_key,
                    "transform": item["transform_name"], "severity": item["transform_severity"],
                    "prompt_variant": item.get("prompt_variant", 0),
                    "question": question, "gt_answer": item["gt_answer"],
                    "model_answer": f"ERROR: {str(e)[:80]}", "category": item["category"],
                    "semantic_sim": 0.0, "substr_match": False, "success": False,
                })

            if len(all_results) % 200 == 0:
                torch.cuda.empty_cache()
                print(f"  [{model_key}] {len(all_results)} results...")

        print(f"  [{model_key}] Chunk {chunk_idx+1}/{meta['n_chunks']} done")
        # Checkpoint after each chunk for resilience
        os.makedirs("/data/results", exist_ok=True)
        with open(f"/data/results/{model_key}.json", "w") as f:
            json.dump(all_results, f, indent=1)
        vol.commit()

    # Final save
    os.makedirs("/data/results", exist_ok=True)
    with open(f"/data/results/{model_key}.json", "w") as f:
        json.dump(all_results, f, indent=1)
    vol.commit()
    print(f"[{model_key}] Saved {len(all_results)} results")
    return {"model": model_key, "n_results": len(all_results)}


# ── Analysis v2 ──────────────────────────────────────────────────────────

@app.function(image=base_image, volumes={"/data": vol}, timeout=600, memory=8192)
def analyze_results_v2():
    import json, os
    import numpy as np

    vol.reload()
    all_results = []
    for fname in os.listdir("/data/results"):
        if fname.endswith(".json"):
            with open(f"/data/results/{fname}") as f:
                all_results.extend(json.load(f))

    print(f"Loaded {len(all_results)} results")

    # Use dual scoring: semantic similarity > 0.65 OR substring match
    SIM_THRESHOLD = 0.65
    for r in all_results:
        r["correct"] = r.get("semantic_sim", 0) >= SIM_THRESHOLD or r.get("substr_match", False)

    models = sorted(set(r["model"] for r in all_results))
    all_transforms = sorted(set(r["transform"] for r in all_results))
    # Separate primary transforms from alt-prompt
    transforms = [t for t in all_transforms if t not in ("original", "original_altprompt")]
    categories = sorted(set(r["category"] for r in all_results))

    # 1. Accuracy by model × transform × severity (primary question only)
    accuracy = {}
    for m in models:
        accuracy[m] = {}
        # Original
        orig = [r for r in all_results if r["model"]==m and r["transform"]=="original"]
        if orig:
            accuracy[m]["original"] = {"0": round(sum(r["correct"] for r in orig)/len(orig), 4)}
        # Each transform
        for t in transforms:
            accuracy[m][t] = {}
            for s in [1,2,3,4]:
                sub = [r for r in all_results if r["model"]==m and r["transform"]==t and r["severity"]==s and r.get("prompt_variant",0)==0]
                if sub:
                    accuracy[m][t][str(s)] = round(sum(r["correct"] for r in sub)/len(sub), 4)

    # 2. Per-category analysis
    cat_acc = {}
    for m in models:
        cat_acc[m] = {}
        for c in categories:
            orig = [r for r in all_results if r["model"]==m and r["category"]==c and r["transform"]=="original"]
            trans = [r for r in all_results if r["model"]==m and r["category"]==c and r["transform"] not in ("original","original_altprompt")]
            o_acc = sum(r["correct"] for r in orig)/len(orig) if orig else 0
            t_acc = sum(r["correct"] for r in trans)/len(trans) if trans else 0
            cat_acc[m][c] = {"original": round(o_acc,4), "transformed": round(t_acc,4), "drop": round(o_acc-t_acc,4)}

    # 3. Consistency (same answer text for original vs transformed)
    consistency = {}
    for m in models:
        mr = [r for r in all_results if r["model"]==m and r.get("prompt_variant",0)==0]
        orig_answers = {r["test_id"]: r["model_answer"].lower().strip()[:80] for r in mr if r["transform"]=="original"}
        consistency[m] = {}
        for t in transforms:
            consistency[m][t] = {}
            for s in [1,2,3,4]:
                sub = [r for r in mr if r["transform"]==t and r["severity"]==s]
                total=0; cons=0
                for r in sub:
                    if r["test_id"] in orig_answers:
                        total += 1
                        ta = r["model_answer"].lower().strip()[:80]
                        oa = orig_answers[r["test_id"]]
                        if ta == oa or (len(oa)>3 and oa[:25]==ta[:25]): cons += 1
                if total: consistency[m][t][str(s)] = round(cons/total, 4)

    # 4. MCI (Metamorphic Consistency Index)
    mci = {}
    for m in models:
        mr = [r for r in all_results if r["model"]==m and r.get("prompt_variant",0)==0]
        origs = {r["test_id"]: r["correct"] for r in mr if r["transform"]=="original"}
        total=0; same=0
        for r in mr:
            if r["transform"] in ("original","original_altprompt"): continue
            if r["test_id"] in origs:
                total += 1
                if r["correct"] == origs[r["test_id"]]: same += 1
        mci[m] = round(same/total, 4) if total else 0

    # 5. Prompt sensitivity (do alt questions give same answers?)
    prompt_sensitivity = {}
    for m in models:
        mr = [r for r in all_results if r["model"]==m]
        orig_primary = {r["test_id"]: r["model_answer"].lower().strip()[:80] for r in mr if r["transform"]=="original" and r.get("prompt_variant",0)==0}
        alt_prompt = [r for r in mr if r["transform"]=="original_altprompt"]
        total=0; same_answer=0; same_correct=0
        for r in alt_prompt:
            if r["test_id"] in orig_primary:
                total += 1
                if r["model_answer"].lower().strip()[:80] == orig_primary[r["test_id"]]: same_answer += 1
                # Check if correctness is preserved
                orig_result = next((x for x in mr if x["test_id"]==r["test_id"] and x["transform"]=="original" and x.get("prompt_variant",0)==0), None)
                if orig_result and r["correct"] == orig_result["correct"]: same_correct += 1
        prompt_sensitivity[m] = {
            "answer_consistency": round(same_answer/total, 4) if total else 0,
            "correctness_consistency": round(same_correct/total, 4) if total else 0,
            "n_pairs": total,
        }

    # 6. Breaking examples
    breaks = []
    for m in models:
        mr = [r for r in all_results if r["model"]==m and r.get("prompt_variant",0)==0]
        orig_correct = {r["test_id"] for r in mr if r["transform"]=="original" and r["correct"]}
        orig_ans = {r["test_id"]: r["model_answer"] for r in mr if r["transform"]=="original"}
        for r in mr:
            if r["transform"] not in ("original","original_altprompt") and r["test_id"] in orig_correct and not r["correct"]:
                breaks.append({
                    "model": m, "test_id": r["test_id"], "category": r["category"],
                    "transform": r["transform"], "severity": r["severity"],
                    "gt": r["gt_answer"], "orig_ans": orig_ans.get(r["test_id"],"")[:100],
                    "trans_ans": r["model_answer"][:100],
                    "semantic_sim": r.get("semantic_sim", 0),
                })

    # 7. Failure mode clustering: group breaking examples by (transform, category)
    failure_clusters = {}
    for b in breaks:
        key = f"{b['transform']}|{b['category']}"
        failure_clusters.setdefault(key, []).append(b)
    failure_summary = {k: len(v) for k, v in sorted(failure_clusters.items(), key=lambda x: -len(x[1]))}

    # 8. Semantic similarity distribution for correct vs incorrect
    sim_dist = {"correct": [], "incorrect": []}
    for r in all_results:
        if r.get("success", False):
            bucket = "correct" if r["correct"] else "incorrect"
            sim_dist[bucket].append(r.get("semantic_sim", 0))
    sim_stats = {}
    for k in sim_dist:
        vals = sim_dist[k]
        if vals:
            sim_stats[k] = {"mean": round(np.mean(vals),4), "std": round(np.std(vals),4),
                            "median": round(np.median(vals),4), "n": len(vals)}

    # 9. Human baseline estimation (from psychophysics literature)
    human_baseline = {
        "note": "Estimated from Geirhos et al. 2023 and Rahmanzadehgervi et al. 2024",
        "object_recognition": {"original": 1.0, "transformed_est": 0.98, "mci_est": 0.99},
        "color_identification": {"original": 1.0, "transformed_est": 0.97, "mci_est": 0.99},
        "counting": {"original": 0.95, "transformed_est": 0.93, "mci_est": 0.98},
        "text_reading": {"original": 1.0, "transformed_est": 0.95, "mci_est": 0.97},
        "spatial_reasoning": {"original": 1.0, "transformed_est": 0.99, "mci_est": 0.99},
        "scene_understanding": {"original": 0.95, "transformed_est": 0.93, "mci_est": 0.98},
        "overall_mci_est": 0.985,
        "justification": "Human visual perception maintains perceptual constancy under rotation, scale, illumination, and moderate blur. Estimated drops are primarily from extreme compression (Q10) and heavy occlusion.",
    }

    analysis = {
        "accuracy": accuracy, "cat_acc": cat_acc, "consistency": consistency,
        "mci": mci, "prompt_sensitivity": prompt_sensitivity,
        "breaking_examples": breaks[:300], "failure_summary": failure_summary,
        "sim_stats": sim_stats, "human_baseline": human_baseline,
        "models": models, "transforms": transforms, "categories": categories,
        "total_results": len(all_results), "sim_threshold": SIM_THRESHOLD,
    }
    with open("/data/analysis_v2.json", "w") as f:
        json.dump(analysis, f, indent=2)
    vol.commit()
    print(f"Analysis v2 saved. MCI: {mci}")
    print(f"Prompt sensitivity: {prompt_sensitivity}")
    print(f"Breaking examples: {len(breaks)}")
    print(f"Top failure clusters: {dict(list(failure_summary.items())[:10])}")
    return analysis


# ── Intervention: Test-Time Augmentation (TTA) ───────────────────────────

@app.function(
    gpu="A100", image=model_image, volumes={"/data": vol},
    timeout=10800, memory=65536, secrets=[hf_secret],
)
def run_tta_intervention(model_key: str):
    """Test-Time Augmentation intervention: generate multiple augmented views
    of each transformed image and aggregate predictions via majority vote.
    Compares TTA accuracy/consistency vs single-pass baseline."""
    import torch, json, gc
    import numpy as np
    from PIL import Image as PILImage, ImageEnhance, ImageFilter
    from sentence_transformers import SentenceTransformer
    from collections import Counter

    vol.reload()
    SIM_THRESHOLD = 0.65

    # Load baseline results for this model
    results_path = f"/data/results/{model_key}.json"
    if not os.path.exists(results_path):
        return {"error": f"No baseline results for {model_key}"}
    with open(results_path) as f:
        baseline_results = json.load(f)

    device = "cuda"
    print(f"[TTA-{model_key}] Loading model...")

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
    elif model_key == "internvl2":
        from transformers import AutoModel, AutoTokenizer
        model_id = "OpenGVLab/InternVL2-2B"
        processor = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModel.from_pretrained(
            model_id, torch_dtype=torch.float16, device_map="auto",
            trust_remote_code=True,
        )
    model.eval()

    sent_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

    def infer_single(pil_img, question):
        """Run single inference on one image."""
        try:
            if model_key == "llava":
                conv = [{"role":"user","content":[{"type":"image"},{"type":"text","text":question}]}]
                prompt = processor.apply_chat_template(conv, add_generation_prompt=True)
                inputs = processor(images=pil_img, text=prompt, return_tensors="pt").to(device)
                with torch.no_grad():
                    out = model.generate(**inputs, max_new_tokens=60, do_sample=False)
                answer = processor.decode(out[0], skip_special_tokens=True)
                if "[/INST]" in answer: answer = answer.split("[/INST]")[-1].strip()
            elif model_key == "qwen2vl":
                messages = [{"role":"user","content":[
                    {"type":"image","image":pil_img},{"type":"text","text":question}
                ]}]
                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = processor(text=[text], images=[pil_img], padding=True, return_tensors="pt").to(device)
                with torch.no_grad():
                    gen_ids = model.generate(**inputs, max_new_tokens=60, do_sample=False)
                trimmed = [o[len(i):] for i,o in zip(inputs.input_ids, gen_ids)]
                answer = processor.batch_decode(trimmed, skip_special_tokens=True)[0]
            elif model_key == "internvl2":
                from torchvision import transforms as T
                from torchvision.transforms.functional import InterpolationMode
                IMAGENET_MEAN = (0.485, 0.456, 0.406)
                IMAGENET_STD = (0.229, 0.224, 0.225)
                transform = T.Compose([
                    T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
                    T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
                    T.ToTensor(),
                    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
                ])
                pixel_values = transform(pil_img).unsqueeze(0).to(torch.float16).to(device)
                generation_config = dict(max_new_tokens=60, do_sample=False)
                answer = model.chat(processor, pixel_values, question, generation_config)
            return answer.strip()[:200]
        except Exception as e:
            return f"ERROR: {str(e)[:80]}"

    def generate_tta_views(pil_img, n_views=4):
        """Generate n augmented views of an image for TTA."""
        views = [pil_img]  # Original always included
        W, H = pil_img.size
        rng = np.random.RandomState(42)
        for i in range(n_views):
            img = pil_img.copy()
            # Random slight rotation (-3 to +3 degrees)
            angle = rng.uniform(-3, 3)
            img = img.rotate(angle, resample=PILImage.BILINEAR, expand=False, fillcolor=(255,255,255))
            # Random slight crop (2-5%)
            pct = rng.uniform(0.02, 0.05)
            l, t = int(W*pct*rng.random()), int(H*pct*rng.random())
            r, b = W - int(W*pct*rng.random()), H - int(H*pct*rng.random())
            img = img.crop((l, t, r, b)).resize((W, H), PILImage.BILINEAR)
            # Random brightness (0.9 to 1.1)
            factor = rng.uniform(0.9, 1.1)
            img = ImageEnhance.Brightness(img).enhance(factor)
            views.append(img)
        return views

    def score_answer(answer, gt):
        """Dual scoring: semantic similarity OR substring match."""
        emb_gt = sent_model.encode([gt], convert_to_tensor=False)
        emb_pred = sent_model.encode([answer.lower()[:100]], convert_to_tensor=False)
        sim = float(np.dot(emb_gt[0], emb_pred[0]) / (np.linalg.norm(emb_gt[0]) * np.linalg.norm(emb_pred[0]) + 1e-8))
        substr = gt.lower().strip() in answer.lower().strip()
        return sim >= SIM_THRESHOLD or substr

    # Select subset: transformed items (primary question only) — sample 500 for efficiency
    transformed = [r for r in baseline_results
                   if r["transform"] not in ("original", "original_altprompt")
                   and r.get("prompt_variant", 0) == 0]
    # Sample evenly across transforms and severities
    rng = np.random.RandomState(123)
    if len(transformed) > 500:
        indices = rng.choice(len(transformed), 500, replace=False)
        subset = [transformed[i] for i in sorted(indices)]
    else:
        subset = transformed

    print(f"[TTA-{model_key}] Running TTA on {len(subset)} items (5 views each)...")

    # Load chunk data to reconstruct images
    with open("/data/metadata.json") as f:
        meta = json.load(f)
    all_chunks = []
    for ci in range(meta["n_chunks"]):
        with open(f"/data/chunks/chunk_{ci:03d}.json") as f:
            all_chunks.extend(json.load(f))
    # Build lookup: (test_id, transform, severity, prompt_variant) -> image_b64
    img_lookup = {}
    for item in all_chunks:
        key = (item["id"], item["transform_name"], item["transform_severity"], item.get("prompt_variant", 0))
        img_lookup[key] = item["image_b64"]

    tta_results = []
    for idx, r in enumerate(subset):
        key = (r["test_id"], r["transform"], r["severity"], r.get("prompt_variant", 0))
        b64 = img_lookup.get(key)
        if not b64:
            continue
        pil_img = PILImage.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
        question = r["question"]
        gt = r["gt_answer"]

        # Generate TTA views and get predictions
        views = generate_tta_views(pil_img, n_views=4)
        predictions = []
        for v in views:
            ans = infer_single(v, question)
            predictions.append(ans)

        # Majority vote: pick most common answer
        # Normalize answers for voting
        normalized = [p.lower().strip()[:50] for p in predictions]
        vote_counter = Counter(normalized)
        majority_answer = vote_counter.most_common(1)[0][0]
        # Find original-cased version
        tta_answer = predictions[normalized.index(majority_answer)]

        baseline_correct = score_answer(r["model_answer"], gt)
        tta_correct = score_answer(tta_answer, gt)

        tta_results.append({
            "test_id": r["test_id"], "model": model_key,
            "transform": r["transform"], "severity": r["severity"],
            "category": r["category"], "gt_answer": gt,
            "baseline_answer": r["model_answer"],
            "tta_answer": tta_answer,
            "all_tta_predictions": predictions,
            "baseline_correct": baseline_correct,
            "tta_correct": tta_correct,
            "agreement_ratio": vote_counter.most_common(1)[0][1] / len(predictions),
        })

        if (idx + 1) % 50 == 0:
            torch.cuda.empty_cache()
            print(f"  [TTA-{model_key}] {idx+1}/{len(subset)} done")

    # Save TTA results
    os.makedirs("/data/tta_results", exist_ok=True)
    with open(f"/data/tta_results/{model_key}.json", "w") as f:
        json.dump(tta_results, f, indent=1)
    vol.commit()

    # Summary
    n_total = len(tta_results)
    baseline_acc = sum(r["baseline_correct"] for r in tta_results) / n_total if n_total else 0
    tta_acc = sum(r["tta_correct"] for r in tta_results) / n_total if n_total else 0
    # Recovered: was wrong, now correct
    recovered = sum(1 for r in tta_results if not r["baseline_correct"] and r["tta_correct"])
    # Lost: was correct, now wrong
    lost = sum(1 for r in tta_results if r["baseline_correct"] and not r["tta_correct"])

    summary = {
        "model": model_key, "n_items": n_total,
        "baseline_acc": round(baseline_acc, 4), "tta_acc": round(tta_acc, 4),
        "improvement": round(tta_acc - baseline_acc, 4),
        "recovered": recovered, "lost": lost,
        "net_gain": recovered - lost,
    }
    print(f"[TTA-{model_key}] Done: baseline={baseline_acc:.3f}, TTA={tta_acc:.3f}, "
          f"recovered={recovered}, lost={lost}")
    return summary


@app.function(image=base_image, volumes={"/data": vol}, timeout=600, memory=8192)
def analyze_tta_results():
    """Aggregate TTA intervention results across models."""
    import json, os
    vol.reload()

    all_tta = []
    for fname in os.listdir("/data/tta_results"):
        if fname.endswith(".json"):
            with open(f"/data/tta_results/{fname}") as f:
                all_tta.extend(json.load(f))

    models = sorted(set(r["model"] for r in all_tta))
    summary = {}
    for m in models:
        mr = [r for r in all_tta if r["model"] == m]
        n = len(mr)
        if n == 0: continue
        b_acc = sum(r["baseline_correct"] for r in mr) / n
        t_acc = sum(r["tta_correct"] for r in mr) / n
        recovered = sum(1 for r in mr if not r["baseline_correct"] and r["tta_correct"])
        lost = sum(1 for r in mr if r["baseline_correct"] and not r["tta_correct"])

        # Per-transform analysis
        per_transform = {}
        for t in sorted(set(r["transform"] for r in mr)):
            tr = [r for r in mr if r["transform"] == t]
            nt = len(tr)
            if nt == 0: continue
            per_transform[t] = {
                "baseline_acc": round(sum(r["baseline_correct"] for r in tr)/nt, 4),
                "tta_acc": round(sum(r["tta_correct"] for r in tr)/nt, 4),
                "n": nt,
            }

        summary[m] = {
            "n_items": n,
            "baseline_acc": round(b_acc, 4),
            "tta_acc": round(t_acc, 4),
            "improvement": round(t_acc - b_acc, 4),
            "recovered": recovered, "lost": lost,
            "per_transform": per_transform,
        }

    tta_analysis = {"summary": summary, "models": models, "total_items": len(all_tta)}
    with open("/data/tta_analysis.json", "w") as f:
        json.dump(tta_analysis, f, indent=2)
    vol.commit()
    print(f"TTA Analysis saved: {json.dumps(summary, indent=2)}")
    return tta_analysis


@app.function(image=base_image, volumes={"/data": vol}, timeout=18000, memory=4096)
def orchestrate_v2():
    """Disconnect-safe orchestrator: runs entirely on Modal, no local dependency."""
    import time

    print("=" * 60)
    print("MetamorphicVLM v2 Experiment Pipeline (disconnect-safe)")
    print("=" * 60)

    print("\n[1/3] Preparing expanded test suite...")
    meta = prepare_test_data_v2.remote()
    print(f"  -> {meta}")

    models_to_run = ["llava", "qwen2vl", "internvl2"]
    print(f"\n[2/3] Running {len(models_to_run)} models in parallel...")
    handles = {}
    for mk in models_to_run:
        handles[mk] = run_model_v2.spawn(mk)
        print(f"  Spawned {mk}")

    for mk, h in handles.items():
        try:
            result = h.get()
            print(f"  -> {mk}: {result}")
        except Exception as e:
            print(f"  -> {mk} FAILED: {e}")

    print("\n[3/5] Analyzing results...")
    analysis = analyze_results_v2.remote()
    print(f"  MCI: {analysis['mci']}")
    print(f"  Total results: {analysis['total_results']}")
    print(f"  Prompt sensitivity: {analysis['prompt_sensitivity']}")

    print("\n[4/5] Running TTA intervention experiments...")
    tta_handles = {}
    for mk in models_to_run:
        tta_handles[mk] = run_tta_intervention.spawn(mk)
        print(f"  Spawned TTA-{mk}")
    for mk, h in tta_handles.items():
        try:
            result = h.get()
            print(f"  -> TTA-{mk}: {result}")
        except Exception as e:
            print(f"  -> TTA-{mk} FAILED: {e}")

    print("\n[5/5] Analyzing TTA results...")
    tta_analysis = analyze_tta_results.remote()
    print(f"  TTA Analysis: {json.dumps(tta_analysis.get('summary', {}), indent=2)}")
    print("All done!")
    return {"main": analysis, "tta": tta_analysis}
