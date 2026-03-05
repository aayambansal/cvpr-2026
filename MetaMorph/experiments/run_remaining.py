#!/usr/bin/env python3
"""Run remaining models that failed or were incomplete."""

import os, io, json, time, base64, requests, numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from pathlib import Path
from collections import defaultdict

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Replace claude-sonnet with working vision model, finish gemini
REMAINING_MODELS = {
    "claude-haiku": "anthropic/claude-3.5-haiku-20241022",
    "gemini-flash": "google/gemini-2.0-flash-001",
    "gemini-pro": "google/gemini-2.5-pro-preview",
}

OUTPUT_DIR = Path("experiments/results")
IMAGE_DIR = Path("experiments/images")

def image_to_base64(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    return f"data:image/png;base64,{b64}"

def query_vlm(model_id, image, question, max_retries=3):
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    data_uri = image_to_base64(image)
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": data_uri}},
            {"type": "text", "text": question + " Be concise."}
        ]}],
        "max_tokens": 150, "temperature": 0.0,
    }
    for attempt in range(max_retries):
        try:
            resp = requests.post(API_URL, headers=headers, json=payload, timeout=60)
            if resp.status_code == 429:
                time.sleep(5 * (attempt + 1)); continue
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(f"  Error ({attempt+1}): {e}")
            if attempt < max_retries - 1: time.sleep(3)
    return "[ERROR]"

def compute_consistency(orig, trans, test):
    o, t = orig.lower().strip(), trans.lower().strip()
    if t == "[error]" or o == "[error]": return None
    if "expected" in test:
        oc = test["expected"].lower() in o
        tc = test["expected"].lower() in t
        return 1.0 if oc == tc and oc else (0.5 if oc == tc else 0.0)
    if "expected_keywords" in test:
        ok = sum(1 for kw in test["expected_keywords"] if kw.lower() in o)
        tk = sum(1 for kw in test["expected_keywords"] if kw.lower() in t)
        total = len(test["expected_keywords"])
        if total == 0: return 1.0
        return 1.0 - abs(ok/total - tk/total)
    ow, tw = set(o.split()), set(t.split())
    if len(ow | tw) == 0: return 1.0
    return len(ow & tw) / len(ow | tw)

def get_transforms():
    transforms = {}
    transforms["resize"] = {
        1: lambda img: img.resize((int(img.width*0.9), int(img.height*0.9)), Image.LANCZOS).resize(img.size, Image.LANCZOS),
        2: lambda img: img.resize((int(img.width*0.75), int(img.height*0.75)), Image.LANCZOS).resize(img.size, Image.LANCZOS),
        3: lambda img: img.resize((int(img.width*0.5), int(img.height*0.5)), Image.LANCZOS).resize(img.size, Image.LANCZOS),
        4: lambda img: img.resize((int(img.width*0.35), int(img.height*0.35)), Image.LANCZOS).resize(img.size, Image.LANCZOS),
        5: lambda img: img.resize((int(img.width*0.25), int(img.height*0.25)), Image.LANCZOS).resize(img.size, Image.LANCZOS),
    }
    def crop_margin(img, pct):
        w, h = img.size; m = int(min(w, h) * pct)
        return img.crop((m, m, w-m, h-m)).resize(img.size, Image.LANCZOS)
    transforms["crop"] = {i: (lambda p: lambda img: crop_margin(img, p))(0.03*i) for i in range(1,6)}
    transforms["rotation"] = {
        1: lambda img: img.rotate(0.5, resample=Image.BICUBIC, expand=False, fillcolor=(255,255,255)),
        2: lambda img: img.rotate(1.0, resample=Image.BICUBIC, expand=False, fillcolor=(255,255,255)),
        3: lambda img: img.rotate(2.0, resample=Image.BICUBIC, expand=False, fillcolor=(255,255,255)),
        4: lambda img: img.rotate(5.0, resample=Image.BICUBIC, expand=False, fillcolor=(255,255,255)),
        5: lambda img: img.rotate(10.0, resample=Image.BICUBIC, expand=False, fillcolor=(255,255,255)),
    }
    def jpeg_compress(img, q):
        buf = io.BytesIO()
        img_rgb = img.convert("RGB")
        img_rgb.save(buf, format="JPEG", quality=q); buf.seek(0)
        return Image.open(buf).copy()
    transforms["jpeg"] = {
        1: lambda img: jpeg_compress(img, 85), 2: lambda img: jpeg_compress(img, 65),
        3: lambda img: jpeg_compress(img, 45), 4: lambda img: jpeg_compress(img, 25),
        5: lambda img: jpeg_compress(img, 10),
    }
    transforms["blur"] = {
        1: lambda img: img.filter(ImageFilter.GaussianBlur(radius=0.5)),
        2: lambda img: img.filter(ImageFilter.GaussianBlur(radius=1.0)),
        3: lambda img: img.filter(ImageFilter.GaussianBlur(radius=2.0)),
        4: lambda img: img.filter(ImageFilter.GaussianBlur(radius=3.5)),
        5: lambda img: img.filter(ImageFilter.GaussianBlur(radius=5.0)),
    }
    def add_border_text(img, sz):
        border = 30 + sz * 5
        new = Image.new("RGB", (img.width + 2*border, img.height + 2*border), "white")
        new.paste(img, (border, border))
        draw = ImageDraw.Draw(new)
        texts = ["Lorem ipsum dolor sit amet", "ADVERTISEMENT - BUY NOW", "Page 1 of 3", "CONFIDENTIAL", "www.example.com"]
        try: font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 10+sz*3)
        except: font = ImageFont.load_default()
        for i, t in enumerate(texts[:sz]):
            draw.text((10, 5+i*20), t, fill="gray", font=font)
        return new.resize(img.size, Image.LANCZOS)
    transforms["border_text"] = {i: (lambda s: lambda img: add_border_text(img, s))(i) for i in range(1,6)}
    return transforms

def build_test_suite():
    """Rebuild the same suite referencing existing images."""
    suite = [
        {"id": "geo_count", "image_path": str(IMAGE_DIR/"geometric.png"), "question": "How many shapes are in this image? Answer with just the number.", "expected": "5", "category": "counting"},
        {"id": "geo_color", "image_path": str(IMAGE_DIR/"geometric.png"), "question": "What colors of shapes do you see? List them briefly.", "expected_keywords": ["red","blue","green","yellow","purple"], "category": "color_recognition"},
        {"id": "doc_revenue", "image_path": str(IMAGE_DIR/"document.png"), "question": "What is the revenue mentioned in this document? Answer briefly.", "expected_keywords": ["4.2","million"], "category": "text_reading"},
        {"id": "doc_growth", "image_path": str(IMAGE_DIR/"document.png"), "question": "What is the growth rate mentioned? Answer briefly.", "expected_keywords": ["12"], "category": "text_reading"},
        {"id": "chart_highest", "image_path": str(IMAGE_DIR/"chart.png"), "question": "Which subject has the highest score? Answer with just the subject name.", "expected_keywords": ["biology"], "category": "chart_reading"},
        {"id": "chart_lowest", "image_path": str(IMAGE_DIR/"chart.png"), "question": "Which subject has the lowest score? Answer with just the subject name.", "expected_keywords": ["math"], "category": "chart_reading"},
        {"id": "nature_objects", "image_path": str(IMAGE_DIR/"nature.png"), "question": "What objects can you see in this scene? List them briefly.", "expected_keywords": ["sun","tree"], "category": "scene_understanding"},
        {"id": "count_circles", "image_path": str(IMAGE_DIR/"counting.png"), "question": "How many red circles are in this image? Answer with just the number.", "expected": "7", "category": "counting"},
        {"id": "spatial_left", "image_path": str(IMAGE_DIR/"spatial.png"), "question": "What shape and color is on the left side of the image? Answer briefly.", "expected_keywords": ["blue","square"], "category": "spatial"},
        {"id": "spatial_right", "image_path": str(IMAGE_DIR/"spatial.png"), "question": "What shape and color is on the right side of the image? Answer briefly.", "expected_keywords": ["red","circle"], "category": "spatial"},
        {"id": "spatial_top", "image_path": str(IMAGE_DIR/"spatial.png"), "question": "What shape is at the top of the image? Answer briefly.", "expected_keywords": ["green","triangle"], "category": "spatial"},
        {"id": "geo_count_100", "image_path": str(IMAGE_DIR/"geometric_100.png"), "question": "How many shapes are in this image? Answer with just the number.", "expected": "5", "category": "counting"},
        {"id": "geo_count_200", "image_path": str(IMAGE_DIR/"geometric_200.png"), "question": "How many shapes are in this image? Answer with just the number.", "expected": "5", "category": "counting"},
        {"id": "count_circles_99", "image_path": str(IMAGE_DIR/"counting_99.png"), "question": "How many red circles are in this image? Answer with just the number.", "expected": "7", "category": "counting"},
    ]
    return suite

def run():
    suite = build_test_suite()
    transforms = get_transforms()
    
    # Load existing results
    existing = []
    partial_path = OUTPUT_DIR / "results_partial.json"
    if partial_path.exists():
        with open(partial_path) as f:
            existing = json.load(f)
    
    # Filter out claude-sonnet (all errors) and incomplete gemini
    existing_clean = [r for r in existing if r["model"] not in ["claude-sonnet", "gemini-flash", "gemini-pro"]]
    
    # Also keep valid gemini-flash results  
    existing_gemini = [r for r in existing if r["model"] == "gemini-flash" and r["consistency"] is not None]
    
    # Track what gemini-flash tests we already have
    done_keys = set()
    for r in existing_gemini:
        done_keys.add((r["model"], r["test_id"], r["transform"], r["severity"]))
    
    all_results = existing_clean + existing_gemini
    
    for model_name, model_id in REMAINING_MODELS.items():
        print(f"\nTesting: {model_name}")
        for test in suite:
            print(f"  Test: {test['id']}")
            img = Image.open(test["image_path"])
            
            # Get original answer
            orig_key = (model_name, test["id"], "original", 0)
            if orig_key not in done_keys or model_name != "gemini-flash":
                orig_answer = query_vlm(model_id, img, test["question"])
                print(f"    Original: {orig_answer[:60]}")
                time.sleep(0.3)
            else:
                # Find existing original answer
                orig_answer = next((r["original_answer"] for r in existing_gemini 
                                   if r["test_id"] == test["id"]), None)
                if not orig_answer:
                    orig_answer = query_vlm(model_id, img, test["question"])
                    time.sleep(0.3)
            
            for tname, sfuncs in transforms.items():
                for sev, tfn in sfuncs.items():
                    key = (model_name, test["id"], tname, sev)
                    if key in done_keys:
                        continue
                    
                    try:
                        timg = tfn(img.copy())
                    except Exception as e:
                        print(f"    Err {tname}/{sev}: {e}"); continue
                    
                    ta = query_vlm(model_id, timg, test["question"])
                    time.sleep(0.3)
                    
                    c = compute_consistency(orig_answer, ta, test)
                    result = {
                        "model": model_name, "test_id": test["id"],
                        "category": test["category"], "question": test["question"],
                        "transform": tname, "severity": sev,
                        "original_answer": orig_answer, "transformed_answer": ta,
                        "consistency": c,
                    }
                    all_results.append(result)
                    
                    if c is not None and c < 0.5:
                        print(f"    !! {tname}/s{sev}: {orig_answer[:30]} vs {ta[:30]}")
        
        # Save after each model
        with open(OUTPUT_DIR / "results_all.json", "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"  Saved {len(all_results)} total results")
    
    print(f"\nDone. Total: {len(all_results)}")

if __name__ == "__main__":
    run()
