#!/usr/bin/env python3
"""
Metamorphic Testing for Vision-Language Model Robustness
=========================================================
Generates test images, applies semantic-preserving transformations,
queries VLMs, and measures answer consistency.
"""

import os
import io
import json
import time
import base64
import hashlib
import requests
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from pathlib import Path
from collections import defaultdict

# ── Configuration ──────────────────────────────────────────────────────────
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
API_URL = "https://openrouter.ai/api/v1/chat/completions"

MODELS = {
    "gpt4o": "openai/gpt-4o",
    "gpt4o-mini": "openai/gpt-4o-mini",
    "claude-sonnet": "anthropic/claude-sonnet-4-20250514",
    "gemini-flash": "google/gemini-2.0-flash-001",
    "gemini-pro": "google/gemini-2.5-pro-preview",
}

OUTPUT_DIR = Path("experiments/results")
IMAGE_DIR = Path("experiments/images")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
IMAGE_DIR.mkdir(parents=True, exist_ok=True)

# ── Image Generation ───────────────────────────────────────────────────────

def create_geometric_scene(seed=42):
    """Scene with colored shapes - tests color/shape recognition."""
    rng = np.random.RandomState(seed)
    img = Image.new("RGB", (512, 512), "white")
    draw = ImageDraw.Draw(img)
    
    shapes = []
    colors = {"red": (220, 50, 50), "blue": (50, 50, 220), 
              "green": (50, 180, 50), "yellow": (220, 200, 50),
              "purple": (150, 50, 200)}
    
    shape_types = ["circle", "rectangle", "triangle"]
    
    for i in range(5):
        x, y = rng.randint(60, 440), rng.randint(60, 440)
        color_name = list(colors.keys())[i % len(colors)]
        color_rgb = colors[color_name]
        stype = shape_types[i % len(shape_types)]
        size = rng.randint(40, 80)
        
        if stype == "circle":
            draw.ellipse([x-size, y-size, x+size, y+size], fill=color_rgb, outline="black", width=2)
        elif stype == "rectangle":
            draw.rectangle([x-size, y-size//2, x+size, y+size//2], fill=color_rgb, outline="black", width=2)
        elif stype == "triangle":
            draw.polygon([(x, y-size), (x-size, y+size), (x+size, y+size)], fill=color_rgb, outline="black", width=2)
        
        shapes.append((stype, color_name))
    
    return img, shapes

def create_text_document(seed=42):
    """Image with readable text - tests OCR capability."""
    img = Image.new("RGB", (512, 400), "white")
    draw = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 24)
        font_small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
    except:
        font = ImageFont.load_default()
        font_small = font
    
    draw.text((30, 30), "QUARTERLY REPORT - Q3 2024", fill="black", font=font)
    draw.line([(30, 65), (480, 65)], fill="black", width=2)
    draw.text((30, 80), "Revenue: $4.2 Million", fill="black", font=font_small)
    draw.text((30, 110), "Operating Costs: $2.8 Million", fill="black", font=font_small)
    draw.text((30, 140), "Net Profit: $1.4 Million", fill="black", font=font_small)
    draw.text((30, 180), "Growth Rate: 12% YoY", fill="black", font=font_small)
    draw.text((30, 220), "Employees: 247", fill="black", font=font_small)
    
    return img

def create_chart_image(seed=42):
    """Simple bar chart image."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(5.12, 5.12), dpi=100)
    categories = ['Physics', 'Chemistry', 'Biology', 'Math', 'English']
    values = [85, 72, 91, 68, 78]
    bars = ax.bar(categories, values, color=['#4C72B0', '#55A868', '#C44E52', '#8172B2', '#CCB974'])
    ax.set_ylabel('Score')
    ax.set_title('Student Test Scores by Subject')
    ax.set_ylim(0, 100)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 2,
                str(val), ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    
    buf = io.BytesIO()
    fig.savefig(buf, format='PNG', dpi=100)
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf).copy()
    buf.close()
    return img

def create_natural_scene(seed=42):
    """Synthetic natural scene with sky, ground, sun, tree."""
    img = Image.new("RGB", (512, 512))
    draw = ImageDraw.Draw(img)
    
    # Sky gradient
    for y in range(300):
        r = int(100 + (155 * y / 300))
        g = int(150 + (105 * y / 300))
        b = 255
        draw.line([(0, y), (512, y)], fill=(r, g, b))
    
    # Ground
    draw.rectangle([0, 300, 512, 512], fill=(34, 139, 34))
    
    # Sun
    draw.ellipse([380, 30, 460, 110], fill=(255, 223, 0))
    
    # Tree trunk
    draw.rectangle([230, 200, 270, 350], fill=(101, 67, 33))
    # Tree crown
    draw.ellipse([180, 120, 320, 260], fill=(0, 100, 0))
    
    # Path
    draw.polygon([(230, 512), (280, 512), (260, 300), (250, 300)], fill=(160, 120, 60))
    
    return img

def create_counting_image(seed=42):
    """Image with countable objects."""
    rng = np.random.RandomState(seed)
    img = Image.new("RGB", (512, 512), (240, 240, 255))
    draw = ImageDraw.Draw(img)
    
    n_circles = 7
    positions = []
    for i in range(n_circles):
        while True:
            x, y = rng.randint(60, 450), rng.randint(60, 450)
            # Check no overlap
            ok = all(((x - px)**2 + (y - py)**2)**0.5 > 70 for px, py in positions)
            if ok or len(positions) == 0:
                positions.append((x, y))
                break
        r = 25
        draw.ellipse([x-r, y-r, x+r, y+r], fill=(220, 50, 50), outline="black", width=2)
    
    return img, n_circles

def create_spatial_image(seed=42):
    """Image testing spatial reasoning - object positions."""
    img = Image.new("RGB", (512, 512), (255, 255, 240))
    draw = ImageDraw.Draw(img)
    
    # Blue square on the left
    draw.rectangle([50, 200, 150, 300], fill=(50, 50, 220), outline="black", width=2)
    # Red circle on the right
    draw.ellipse([350, 200, 450, 300], fill=(220, 50, 50), outline="black", width=2)
    # Green triangle on top center
    draw.polygon([(256, 50), (206, 150), (306, 150)], fill=(50, 180, 50), outline="black", width=2)
    
    return img

# ── Test Suite ─────────────────────────────────────────────────────────────

def build_test_suite():
    """Build diverse test images with questions and expected answers."""
    suite = []
    
    # 1. Geometric scene
    img, shapes = create_geometric_scene()
    img.save(IMAGE_DIR / "geometric.png")
    suite.append({
        "id": "geo_count",
        "image_path": str(IMAGE_DIR / "geometric.png"),
        "question": "How many shapes are in this image? Answer with just the number.",
        "expected": "5",
        "category": "counting"
    })
    suite.append({
        "id": "geo_color",
        "image_path": str(IMAGE_DIR / "geometric.png"),
        "question": "What colors of shapes do you see? List them briefly.",
        "expected_keywords": ["red", "blue", "green", "yellow", "purple"],
        "category": "color_recognition"
    })
    
    # 2. Text document
    img = create_text_document()
    img.save(IMAGE_DIR / "document.png")
    suite.append({
        "id": "doc_revenue",
        "image_path": str(IMAGE_DIR / "document.png"),
        "question": "What is the revenue mentioned in this document? Answer briefly.",
        "expected_keywords": ["4.2", "million"],
        "category": "text_reading"
    })
    suite.append({
        "id": "doc_growth",
        "image_path": str(IMAGE_DIR / "document.png"),
        "question": "What is the growth rate mentioned? Answer briefly.",
        "expected_keywords": ["12"],
        "category": "text_reading"
    })
    
    # 3. Chart
    img = create_chart_image()
    img.save(IMAGE_DIR / "chart.png")
    suite.append({
        "id": "chart_highest",
        "image_path": str(IMAGE_DIR / "chart.png"),
        "question": "Which subject has the highest score? Answer with just the subject name.",
        "expected_keywords": ["biology"],
        "category": "chart_reading"
    })
    suite.append({
        "id": "chart_lowest",
        "image_path": str(IMAGE_DIR / "chart.png"),
        "question": "Which subject has the lowest score? Answer with just the subject name.",
        "expected_keywords": ["math"],
        "category": "chart_reading"
    })
    
    # 4. Natural scene
    img = create_natural_scene()
    img.save(IMAGE_DIR / "nature.png")
    suite.append({
        "id": "nature_objects",
        "image_path": str(IMAGE_DIR / "nature.png"),
        "question": "What objects can you see in this scene? List them briefly.",
        "expected_keywords": ["sun", "tree"],
        "category": "scene_understanding"
    })
    
    # 5. Counting
    img, count = create_counting_image()
    img.save(IMAGE_DIR / "counting.png")
    suite.append({
        "id": "count_circles",
        "image_path": str(IMAGE_DIR / "counting.png"),
        "question": "How many red circles are in this image? Answer with just the number.",
        "expected": "7",
        "category": "counting"
    })
    
    # 6. Spatial reasoning
    img = create_spatial_image()
    img.save(IMAGE_DIR / "spatial.png")
    suite.append({
        "id": "spatial_left",
        "image_path": str(IMAGE_DIR / "spatial.png"),
        "question": "What shape and color is on the left side of the image? Answer briefly.",
        "expected_keywords": ["blue", "square"],
        "category": "spatial"
    })
    suite.append({
        "id": "spatial_right",
        "image_path": str(IMAGE_DIR / "spatial.png"),
        "question": "What shape and color is on the right side of the image? Answer briefly.",
        "expected_keywords": ["red", "circle"],
        "category": "spatial"
    })
    suite.append({
        "id": "spatial_top",
        "image_path": str(IMAGE_DIR / "spatial.png"),
        "question": "What shape is at the top of the image? Answer briefly.",
        "expected_keywords": ["green", "triangle"],
        "category": "spatial"
    })
    
    # Additional scene variants with different seeds
    for seed in [100, 200]:
        img2, shapes2 = create_geometric_scene(seed=seed)
        img2.save(IMAGE_DIR / f"geometric_{seed}.png")
        suite.append({
            "id": f"geo_count_{seed}",
            "image_path": str(IMAGE_DIR / f"geometric_{seed}.png"),
            "question": "How many shapes are in this image? Answer with just the number.",
            "expected": "5",
            "category": "counting"
        })
    
    img3, count3 = create_counting_image(seed=99)
    img3.save(IMAGE_DIR / "counting_99.png")
    suite.append({
        "id": "count_circles_99",
        "image_path": str(IMAGE_DIR / "counting_99.png"),
        "question": "How many red circles are in this image? Answer with just the number.",
        "expected": "7",
        "category": "counting"
    })
    
    return suite

# ── Transformations ────────────────────────────────────────────────────────

def get_transformations():
    """Return transformation functions at multiple severity levels."""
    transforms = {}
    
    # 1. Resize
    transforms["resize"] = {
        1: lambda img: img.resize((int(img.width*0.9), int(img.height*0.9)), Image.LANCZOS).resize(img.size, Image.LANCZOS),
        2: lambda img: img.resize((int(img.width*0.75), int(img.height*0.75)), Image.LANCZOS).resize(img.size, Image.LANCZOS),
        3: lambda img: img.resize((int(img.width*0.5), int(img.height*0.5)), Image.LANCZOS).resize(img.size, Image.LANCZOS),
        4: lambda img: img.resize((int(img.width*0.35), int(img.height*0.35)), Image.LANCZOS).resize(img.size, Image.LANCZOS),
        5: lambda img: img.resize((int(img.width*0.25), int(img.height*0.25)), Image.LANCZOS).resize(img.size, Image.LANCZOS),
    }
    
    # 2. Crop margins (center crop, then resize back)
    def crop_margin(img, pct):
        w, h = img.size
        m = int(min(w, h) * pct)
        cropped = img.crop((m, m, w-m, h-m))
        return cropped.resize(img.size, Image.LANCZOS)
    
    transforms["crop"] = {
        1: lambda img: crop_margin(img, 0.03),
        2: lambda img: crop_margin(img, 0.06),
        3: lambda img: crop_margin(img, 0.10),
        4: lambda img: crop_margin(img, 0.15),
        5: lambda img: crop_margin(img, 0.20),
    }
    
    # 3. Rotation (small angles)
    transforms["rotation"] = {
        1: lambda img: img.rotate(0.5, resample=Image.BICUBIC, expand=False, fillcolor=(255,255,255)),
        2: lambda img: img.rotate(1.0, resample=Image.BICUBIC, expand=False, fillcolor=(255,255,255)),
        3: lambda img: img.rotate(2.0, resample=Image.BICUBIC, expand=False, fillcolor=(255,255,255)),
        4: lambda img: img.rotate(5.0, resample=Image.BICUBIC, expand=False, fillcolor=(255,255,255)),
        5: lambda img: img.rotate(10.0, resample=Image.BICUBIC, expand=False, fillcolor=(255,255,255)),
    }
    
    # 4. JPEG compression
    def jpeg_compress(img, quality):
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        return Image.open(buf).copy()
    
    transforms["jpeg"] = {
        1: lambda img: jpeg_compress(img, 85),
        2: lambda img: jpeg_compress(img, 65),
        3: lambda img: jpeg_compress(img, 45),
        4: lambda img: jpeg_compress(img, 25),
        5: lambda img: jpeg_compress(img, 10),
    }
    
    # 5. Gaussian blur
    transforms["blur"] = {
        1: lambda img: img.filter(ImageFilter.GaussianBlur(radius=0.5)),
        2: lambda img: img.filter(ImageFilter.GaussianBlur(radius=1.0)),
        3: lambda img: img.filter(ImageFilter.GaussianBlur(radius=2.0)),
        4: lambda img: img.filter(ImageFilter.GaussianBlur(radius=3.5)),
        5: lambda img: img.filter(ImageFilter.GaussianBlur(radius=5.0)),
    }
    
    # 6. Border text (irrelevant text added to border)
    def add_border_text(img, text_size):
        border = 30 + text_size * 5
        new_img = Image.new("RGB", (img.width + 2*border, img.height + 2*border), "white")
        new_img.paste(img, (border, border))
        draw = ImageDraw.Draw(new_img)
        
        texts = [
            "Lorem ipsum dolor sit amet",
            "ADVERTISEMENT - BUY NOW",
            "Page 1 of 3 | Printed 2024",
            "CONFIDENTIAL - DO NOT COPY",
            "www.example.com | Contact Us"
        ]
        
        try:
            font_size = 10 + text_size * 3
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
        except:
            font = ImageFont.load_default()
        
        for i, t in enumerate(texts[:text_size]):
            draw.text((10, 5 + i*20), t, fill="gray", font=font)
            draw.text((10, img.height + border + 10 + i*20), t, fill="gray", font=font)
        
        # Resize back to original
        return new_img.resize(img.size, Image.LANCZOS)
    
    transforms["border_text"] = {
        1: lambda img: add_border_text(img, 1),
        2: lambda img: add_border_text(img, 2),
        3: lambda img: add_border_text(img, 3),
        4: lambda img: add_border_text(img, 4),
        5: lambda img: add_border_text(img, 5),
    }
    
    return transforms

# ── VLM Query ──────────────────────────────────────────────────────────────

def image_to_base64(img):
    """Convert PIL Image to base64 data URI."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    return f"data:image/png;base64,{b64}"

def query_vlm(model_id, image, question, max_retries=3):
    """Query a VLM with an image and question via OpenRouter."""
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    
    data_uri = image_to_base64(image)
    
    payload = {
        "model": model_id,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_uri}},
                    {"type": "text", "text": question + " Be concise."}
                ]
            }
        ],
        "max_tokens": 150,
        "temperature": 0.0,
    }
    
    for attempt in range(max_retries):
        try:
            resp = requests.post(API_URL, headers=headers, json=payload, timeout=60)
            if resp.status_code == 429:
                wait_time = 5 * (attempt + 1)
                print(f"  Rate limited, waiting {wait_time}s...")
                time.sleep(wait_time)
                continue
            resp.raise_for_status()
            result = resp.json()
            return result["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(f"  Error (attempt {attempt+1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(3)
    return "[ERROR]"

# ── Consistency Scoring ────────────────────────────────────────────────────

def compute_consistency(original_answer, transformed_answer, test_case):
    """Score consistency between original and transformed answers."""
    orig = original_answer.lower().strip()
    trans = transformed_answer.lower().strip()
    
    if trans == "[error]" or orig == "[error]":
        return None  # Skip errors
    
    # Exact match (for numeric answers)
    if "expected" in test_case:
        orig_correct = test_case["expected"].lower() in orig
        trans_correct = test_case["expected"].lower() in trans
        return 1.0 if orig_correct == trans_correct and orig_correct else (0.5 if orig_correct == trans_correct else 0.0)
    
    # Keyword match (for descriptive answers)
    if "expected_keywords" in test_case:
        orig_keywords = sum(1 for kw in test_case["expected_keywords"] if kw.lower() in orig)
        trans_keywords = sum(1 for kw in test_case["expected_keywords"] if kw.lower() in trans)
        total = len(test_case["expected_keywords"])
        if total == 0:
            return 1.0
        orig_score = orig_keywords / total
        trans_score = trans_keywords / total
        # Consistency = both get similar scores
        return 1.0 - abs(orig_score - trans_score)
    
    # Fallback: simple text similarity
    orig_words = set(orig.split())
    trans_words = set(trans.split())
    if len(orig_words | trans_words) == 0:
        return 1.0
    jaccard = len(orig_words & trans_words) / len(orig_words | trans_words)
    return jaccard

# ── Main Experiment ────────────────────────────────────────────────────────

def run_experiments():
    """Run the full metamorphic testing experiment."""
    print("=" * 60)
    print("METAMORPHIC TESTING FOR VLM ROBUSTNESS")
    print("=" * 60)
    
    # Build test suite
    print("\n[1/4] Building test suite...")
    test_suite = build_test_suite()
    print(f"  Created {len(test_suite)} test cases")
    
    # Get transformations
    transforms = get_transformations()
    print(f"  Defined {len(transforms)} transformation types with 5 severity levels each")
    
    # Results storage
    all_results = []
    
    # For each model
    for model_name, model_id in MODELS.items():
        print(f"\n[2/4] Testing model: {model_name} ({model_id})")
        
        for test in test_suite:
            print(f"  Test: {test['id']}")
            img = Image.open(test["image_path"])
            
            # Get original answer
            print(f"    Original query...")
            orig_answer = query_vlm(model_id, img, test["question"])
            print(f"    Answer: {orig_answer[:80]}...")
            time.sleep(0.5)  # Rate limit respect
            
            # Apply each transformation at each severity
            for transform_name, severity_funcs in transforms.items():
                for severity, transform_fn in severity_funcs.items():
                    try:
                        transformed_img = transform_fn(img.copy())
                    except Exception as e:
                        print(f"    Transform error ({transform_name}, sev={severity}): {e}")
                        continue
                    
                    print(f"    {transform_name} (severity={severity})...")
                    trans_answer = query_vlm(model_id, transformed_img, test["question"])
                    time.sleep(0.5)
                    
                    consistency = compute_consistency(orig_answer, trans_answer, test)
                    
                    result = {
                        "model": model_name,
                        "test_id": test["id"],
                        "category": test["category"],
                        "question": test["question"],
                        "transform": transform_name,
                        "severity": severity,
                        "original_answer": orig_answer,
                        "transformed_answer": trans_answer,
                        "consistency": consistency,
                    }
                    all_results.append(result)
                    
                    if consistency is not None and consistency < 0.5:
                        print(f"    !! INCONSISTENCY: {orig_answer[:40]} vs {trans_answer[:40]}")
            
            # Save intermediate results
            with open(OUTPUT_DIR / "results_partial.json", "w") as f:
                json.dump(all_results, f, indent=2)
    
    # Save final results
    print("\n[3/4] Saving results...")
    with open(OUTPUT_DIR / "results_final.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    # Compute summary statistics
    print("\n[4/4] Computing summary statistics...")
    summary = compute_summary(all_results)
    with open(OUTPUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print(f"Total queries: {len(all_results) + len(test_suite) * len(MODELS)}")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("=" * 60)
    
    return all_results, summary

def compute_summary(results):
    """Compute aggregate statistics."""
    summary = {
        "by_model_transform": defaultdict(lambda: defaultdict(list)),
        "by_model_severity": defaultdict(lambda: defaultdict(list)),
        "by_model_category": defaultdict(lambda: defaultdict(list)),
        "by_transform_severity": defaultdict(lambda: defaultdict(list)),
        "overall_by_model": defaultdict(list),
    }
    
    for r in results:
        if r["consistency"] is None:
            continue
        model = r["model"]
        transform = r["transform"]
        severity = r["severity"]
        category = r["category"]
        c = r["consistency"]
        
        summary["by_model_transform"][model][transform].append(c)
        summary["by_model_severity"][model][severity].append(c)
        summary["by_model_category"][model][category].append(c)
        summary["by_transform_severity"][transform][severity].append(c)
        summary["overall_by_model"][model].append(c)
    
    # Compute means
    def mean_dict(d):
        return {k: {k2: round(np.mean(v2), 4) for k2, v2 in v.items()} for k, v in d.items()}
    
    def mean_simple(d):
        return {k: round(np.mean(v), 4) for k, v in d.items()}
    
    return {
        "by_model_transform": mean_dict(dict(summary["by_model_transform"])),
        "by_model_severity": mean_dict(dict(summary["by_model_severity"])),
        "by_model_category": mean_dict(dict(summary["by_model_category"])),
        "by_transform_severity": mean_dict(dict(summary["by_transform_severity"])),
        "overall_by_model": mean_simple(dict(summary["overall_by_model"])),
        "total_tests": len(results),
        "total_valid": len([r for r in results if r["consistency"] is not None]),
    }

if __name__ == "__main__":
    run_experiments()
