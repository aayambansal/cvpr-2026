#!/usr/bin/env python3
"""
CounterBench: Counterfactual Consistency Benchmark for Vision-Language Models.

Generates paired synthetic scenes (original + intervened) with ground-truth
annotations for evaluating counterfactual reasoning in VLMs.

Scene Categories:
  1. SPATIAL   - relational reasoning (left/right, above/below, inside/outside)
  2. CAUSAL    - cause-effect reasoning (arrows, containers, spillage)
  3. COMPOSITIONAL - attribute binding (color+shape combinations)
  4. COUNTING  - numerical reasoning (how many objects)
  5. OCCLUSION - visibility reasoning (partially hidden objects)

Intervention Types:
  - REMOVE:  delete an object
  - REPLACE: change an attribute (color, shape)
  - SWAP:    swap positions of two objects
  - ADD:     introduce a new object

For each pair, we record:
  - question (natural language)
  - answer_original (ground truth for original image)
  - answer_intervened (ground truth for intervened image)
  - should_change (bool: does the correct answer change?)
"""

import json
import os
import random
import math
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

random.seed(42)

# ── Config ────────────────────────────────────────────────────────────────
IMG_SIZE = (512, 512)
BG_COLOR = (245, 245, 245)
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data"
IMG_DIR = OUTPUT_DIR / "images"
IMG_DIR.mkdir(parents=True, exist_ok=True)

COLORS = {
    "red":    (220, 50, 50),
    "blue":   (50, 80, 220),
    "green":  (50, 180, 80),
    "yellow": (230, 200, 40),
    "purple": (140, 50, 180),
    "orange": (240, 140, 30),
}
COLOR_NAMES = list(COLORS.keys())

SHAPES = ["square", "circle", "triangle", "diamond"]

# ── Drawing primitives ────────────────────────────────────────────────────

def draw_shape(draw, shape, color_name, cx, cy, size=50):
    """Draw a named shape centered at (cx, cy)."""
    rgb = COLORS[color_name]
    s = size
    if shape == "square":
        draw.rectangle([cx - s, cy - s, cx + s, cy + s], fill=rgb, outline=(0, 0, 0), width=2)
    elif shape == "circle":
        draw.ellipse([cx - s, cy - s, cx + s, cy + s], fill=rgb, outline=(0, 0, 0), width=2)
    elif shape == "triangle":
        pts = [(cx, cy - s), (cx - s, cy + s), (cx + s, cy + s)]
        draw.polygon(pts, fill=rgb, outline=(0, 0, 0), width=2)
    elif shape == "diamond":
        pts = [(cx, cy - s), (cx + s, cy), (cx, cy + s), (cx - s, cy)]
        draw.polygon(pts, fill=rgb, outline=(0, 0, 0), width=2)


def draw_arrow(draw, x1, y1, x2, y2, color=(60, 60, 60), width=3):
    """Draw an arrow from (x1,y1) to (x2,y2)."""
    draw.line([(x1, y1), (x2, y2)], fill=color, width=width)
    angle = math.atan2(y2 - y1, x2 - x1)
    head_len = 15
    for da in [2.5, -2.5]:
        hx = x2 - head_len * math.cos(angle + da)
        hy = y2 - head_len * math.sin(angle + da)
        draw.line([(x2, y2), (int(hx), int(hy))], fill=color, width=width)


def draw_container(draw, cx, cy, w, h, color=(180, 180, 180)):
    """Draw an open-top container (U-shape)."""
    x0, y0 = cx - w // 2, cy - h // 2
    x1, y1 = cx + w // 2, cy + h // 2
    draw.line([(x0, y0), (x0, y1)], fill=color, width=4)
    draw.line([(x0, y1), (x1, y1)], fill=color, width=4)
    draw.line([(x1, y1), (x1, y0)], fill=color, width=4)


def add_label(draw, text, x, y, font_size=16):
    """Add a small text label."""
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
    except:
        font = ImageFont.load_default()
    draw.text((x, y), text, fill=(40, 40, 40), font=font)


# ── Scene generators ──────────────────────────────────────────────────────

def gen_spatial_leftright(idx):
    """Two objects side by side; question: is A left of B?"""
    pairs = []
    c1, c2 = random.sample(COLOR_NAMES, 2)
    s1, s2 = random.sample(SHAPES, 2)

    # Original: s1 on left, s2 on right
    img = Image.new("RGB", IMG_SIZE, BG_COLOR)
    d = ImageDraw.Draw(img)
    draw_shape(d, s1, c1, 160, 256, 55)
    draw_shape(d, s2, c2, 352, 256, 55)
    add_label(d, "A", 145, 320)
    add_label(d, "B", 337, 320)
    orig_path = f"spatial_lr_{idx:04d}_orig.png"
    img.save(IMG_DIR / orig_path)

    # Intervention: SWAP positions
    img2 = Image.new("RGB", IMG_SIZE, BG_COLOR)
    d2 = ImageDraw.Draw(img2)
    draw_shape(d2, s2, c2, 160, 256, 55)
    draw_shape(d2, s1, c1, 352, 256, 55)
    add_label(d2, "A", 337, 320)
    add_label(d2, "B", 145, 320)
    int_path = f"spatial_lr_{idx:04d}_intv.png"
    img2.save(IMG_DIR / int_path)

    obj_a = f"{c1} {s1}"
    obj_b = f"{c2} {s2}"
    pairs.append({
        "id": f"spatial_lr_{idx:04d}",
        "category": "spatial",
        "subcategory": "left_right",
        "intervention": "swap",
        "original_image": orig_path,
        "intervened_image": int_path,
        "question": f"Is A (the {obj_a}) to the left of B (the {obj_b})?",
        "answer_original": "yes",
        "answer_intervened": "no",
        "should_change": True,
    })
    return pairs


def gen_spatial_above_below(idx):
    """Two objects stacked; question: is A above B?"""
    pairs = []
    c1, c2 = random.sample(COLOR_NAMES, 2)
    s1, s2 = random.sample(SHAPES, 2)

    img = Image.new("RGB", IMG_SIZE, BG_COLOR)
    d = ImageDraw.Draw(img)
    draw_shape(d, s1, c1, 256, 150, 55)
    draw_shape(d, s2, c2, 256, 362, 55)
    add_label(d, "A", 280, 135)
    add_label(d, "B", 280, 347)
    orig_path = f"spatial_ab_{idx:04d}_orig.png"
    img.save(IMG_DIR / orig_path)

    # Intervention: swap vertical positions
    img2 = Image.new("RGB", IMG_SIZE, BG_COLOR)
    d2 = ImageDraw.Draw(img2)
    draw_shape(d2, s2, c2, 256, 150, 55)
    draw_shape(d2, s1, c1, 256, 362, 55)
    add_label(d2, "A", 280, 347)
    add_label(d2, "B", 280, 135)
    int_path = f"spatial_ab_{idx:04d}_intv.png"
    img2.save(IMG_DIR / int_path)

    obj_a = f"{c1} {s1}"
    obj_b = f"{c2} {s2}"
    pairs.append({
        "id": f"spatial_ab_{idx:04d}",
        "category": "spatial",
        "subcategory": "above_below",
        "intervention": "swap",
        "original_image": orig_path,
        "intervened_image": int_path,
        "question": f"Is A (the {obj_a}) above B (the {obj_b})?",
        "answer_original": "yes",
        "answer_intervened": "no",
        "should_change": True,
    })
    return pairs


def gen_spatial_inside(idx):
    """Object inside a container; question: is the object inside the container?"""
    pairs = []
    c1 = random.choice(COLOR_NAMES)
    s1 = random.choice(SHAPES)

    # Original: object inside container
    img = Image.new("RGB", IMG_SIZE, BG_COLOR)
    d = ImageDraw.Draw(img)
    draw_container(d, 256, 280, 180, 160)
    draw_shape(d, s1, c1, 256, 300, 40)
    add_label(d, "container", 200, 180)
    orig_path = f"spatial_in_{idx:04d}_orig.png"
    img.save(IMG_DIR / orig_path)

    # Intervention: move object outside
    img2 = Image.new("RGB", IMG_SIZE, BG_COLOR)
    d2 = ImageDraw.Draw(img2)
    draw_container(d2, 256, 280, 180, 160)
    draw_shape(d2, s1, c1, 440, 160, 40)
    add_label(d2, "container", 200, 180)
    int_path = f"spatial_in_{idx:04d}_intv.png"
    img2.save(IMG_DIR / int_path)

    obj = f"{c1} {s1}"
    pairs.append({
        "id": f"spatial_in_{idx:04d}",
        "category": "spatial",
        "subcategory": "inside_outside",
        "intervention": "move",
        "original_image": orig_path,
        "intervened_image": int_path,
        "question": f"Is the {obj} inside the container?",
        "answer_original": "yes",
        "answer_intervened": "no",
        "should_change": True,
    })
    return pairs


def gen_causal_arrow(idx):
    """Arrow points from A to B suggesting causation; remove arrow."""
    pairs = []
    c1, c2 = random.sample(COLOR_NAMES, 2)
    s1, s2 = random.sample(SHAPES, 2)

    # Original: A --arrow--> B
    img = Image.new("RGB", IMG_SIZE, BG_COLOR)
    d = ImageDraw.Draw(img)
    draw_shape(d, s1, c1, 140, 256, 50)
    draw_shape(d, s2, c2, 380, 256, 50)
    draw_arrow(d, 200, 256, 320, 256)
    add_label(d, "A", 125, 315)
    add_label(d, "B", 365, 315)
    orig_path = f"causal_arrow_{idx:04d}_orig.png"
    img.save(IMG_DIR / orig_path)

    # Intervention: remove arrow
    img2 = Image.new("RGB", IMG_SIZE, BG_COLOR)
    d2 = ImageDraw.Draw(img2)
    draw_shape(d2, s1, c1, 140, 256, 50)
    draw_shape(d2, s2, c2, 380, 256, 50)
    add_label(d2, "A", 125, 315)
    add_label(d2, "B", 365, 315)
    int_path = f"causal_arrow_{idx:04d}_intv.png"
    img2.save(IMG_DIR / int_path)

    obj_a = f"{c1} {s1}"
    obj_b = f"{c2} {s2}"
    pairs.append({
        "id": f"causal_arrow_{idx:04d}",
        "category": "causal",
        "subcategory": "arrow_cause",
        "intervention": "remove_arrow",
        "original_image": orig_path,
        "intervened_image": int_path,
        "question": f"Does the scene show that A (the {obj_a}) causes an effect on B (the {obj_b})?",
        "answer_original": "yes",
        "answer_intervened": "no",
        "should_change": True,
    })
    return pairs


def gen_causal_spill(idx):
    """Tilted container with liquid spill; question: what caused the spill?"""
    pairs = []
    liquid_color = random.choice(["red", "blue", "green"])

    # Original: tilted container with spill
    img = Image.new("RGB", IMG_SIZE, BG_COLOR)
    d = ImageDraw.Draw(img)
    # Tilted container
    pts = [(180, 200), (220, 180), (320, 280), (280, 300)]
    d.polygon(pts, outline=(100, 100, 100), width=3)
    # Liquid spilling
    spill_pts = [(300, 290), (340, 350), (380, 400), (300, 380), (260, 350)]
    d.polygon(spill_pts, fill=COLORS[liquid_color])
    # Ball nearby
    c_ball = random.choice([c for c in COLOR_NAMES if c != liquid_color])
    draw_shape(d, "circle", c_ball, 160, 340, 35)
    add_label(d, "tilted cup", 200, 160)
    add_label(d, "ball", 130, 385)
    orig_path = f"causal_spill_{idx:04d}_orig.png"
    img.save(IMG_DIR / orig_path)

    # Intervention: upright container, no spill
    img2 = Image.new("RGB", IMG_SIZE, BG_COLOR)
    d2 = ImageDraw.Draw(img2)
    draw_container(d2, 250, 260, 120, 140)
    # Liquid inside container
    d2.rectangle([195, 260, 305, 330], fill=COLORS[liquid_color])
    draw_shape(d2, "circle", c_ball, 160, 340, 35)
    add_label(d2, "upright cup", 200, 160)
    add_label(d2, "ball", 130, 385)
    int_path = f"causal_spill_{idx:04d}_intv.png"
    img2.save(IMG_DIR / int_path)

    pairs.append({
        "id": f"causal_spill_{idx:04d}",
        "category": "causal",
        "subcategory": "spill_cause",
        "intervention": "upright_container",
        "original_image": orig_path,
        "intervened_image": int_path,
        "question": f"Is the {liquid_color} liquid spilling out of the container?",
        "answer_original": "yes",
        "answer_intervened": "no",
        "should_change": True,
    })
    return pairs


def gen_causal_chain(idx):
    """A -> B -> C chain; remove middle link."""
    pairs = []
    c1, c2, c3 = random.sample(COLOR_NAMES, 3)
    s1, s2, s3 = random.sample(SHAPES, 3)

    # Original: A -> B -> C
    img = Image.new("RGB", IMG_SIZE, BG_COLOR)
    d = ImageDraw.Draw(img)
    draw_shape(d, s1, c1, 100, 256, 45)
    draw_shape(d, s2, c2, 256, 256, 45)
    draw_shape(d, s3, c3, 412, 256, 45)
    draw_arrow(d, 155, 256, 200, 256)
    draw_arrow(d, 310, 256, 358, 256)
    add_label(d, "A", 85, 310)
    add_label(d, "B", 241, 310)
    add_label(d, "C", 397, 310)
    orig_path = f"causal_chain_{idx:04d}_orig.png"
    img.save(IMG_DIR / orig_path)

    # Intervention: remove B (break chain)
    img2 = Image.new("RGB", IMG_SIZE, BG_COLOR)
    d2 = ImageDraw.Draw(img2)
    draw_shape(d2, s1, c1, 100, 256, 45)
    draw_shape(d2, s3, c3, 412, 256, 45)
    add_label(d2, "A", 85, 310)
    add_label(d2, "C", 397, 310)
    int_path = f"causal_chain_{idx:04d}_intv.png"
    img2.save(IMG_DIR / int_path)

    pairs.append({
        "id": f"causal_chain_{idx:04d}",
        "category": "causal",
        "subcategory": "causal_chain",
        "intervention": "remove_mediator",
        "original_image": orig_path,
        "intervened_image": int_path,
        "question": f"Is there a causal path from A (the {c1} {s1}) to C (the {c3} {s3})?",
        "answer_original": "yes",
        "answer_intervened": "no",
        "should_change": True,
    })
    return pairs


def gen_compositional_attribute(idx):
    """Multiple objects; question: is there a [color] [shape]?"""
    pairs = []
    n_objs = random.randint(3, 5)
    objs = []
    positions = [(130, 170), (380, 170), (130, 370), (380, 370), (256, 256)]
    for i in range(n_objs):
        objs.append((random.choice(SHAPES), random.choice(COLOR_NAMES), positions[i]))

    # Pick a target that exists
    target_idx = random.randint(0, n_objs - 1)
    target_shape, target_color, _ = objs[target_idx]

    # Original
    img = Image.new("RGB", IMG_SIZE, BG_COLOR)
    d = ImageDraw.Draw(img)
    for sh, co, (px, py) in objs:
        draw_shape(d, sh, co, px, py, 45)
    orig_path = f"comp_attr_{idx:04d}_orig.png"
    img.save(IMG_DIR / orig_path)

    # Intervention: change target's color to something different
    new_color = random.choice([c for c in COLOR_NAMES if c != target_color])
    objs_mod = list(objs)
    objs_mod[target_idx] = (target_shape, new_color, objs[target_idx][2])

    img2 = Image.new("RGB", IMG_SIZE, BG_COLOR)
    d2 = ImageDraw.Draw(img2)
    for sh, co, (px, py) in objs_mod:
        draw_shape(d2, sh, co, px, py, 45)
    int_path = f"comp_attr_{idx:04d}_intv.png"
    img2.save(IMG_DIR / int_path)

    # Check if the original combo still exists after intervention
    orig_combos = {(s, c) for s, c, _ in objs}
    new_combos = {(s, c) for s, c, _ in objs_mod}
    target_in_new = (target_shape, target_color) in new_combos

    pairs.append({
        "id": f"comp_attr_{idx:04d}",
        "category": "compositional",
        "subcategory": "attribute_binding",
        "intervention": "replace_color",
        "original_image": orig_path,
        "intervened_image": int_path,
        "question": f"Is there a {target_color} {target_shape} in the scene?",
        "answer_original": "yes",
        "answer_intervened": "yes" if target_in_new else "no",
        "should_change": not target_in_new,
    })
    return pairs


def gen_compositional_relative(idx):
    """Compositional: [color] [shape] above [color] [shape]."""
    pairs = []
    c1, c2 = random.sample(COLOR_NAMES, 2)
    s1, s2 = random.sample(SHAPES, 2)

    # Original: c1-s1 above c2-s2
    img = Image.new("RGB", IMG_SIZE, BG_COLOR)
    d = ImageDraw.Draw(img)
    draw_shape(d, s1, c1, 256, 160, 55)
    draw_shape(d, s2, c2, 256, 360, 55)
    orig_path = f"comp_rel_{idx:04d}_orig.png"
    img.save(IMG_DIR / orig_path)

    # Intervention: change color of top shape
    c3 = random.choice([c for c in COLOR_NAMES if c != c1 and c != c2])
    img2 = Image.new("RGB", IMG_SIZE, BG_COLOR)
    d2 = ImageDraw.Draw(img2)
    draw_shape(d2, s1, c3, 256, 160, 55)
    draw_shape(d2, s2, c2, 256, 360, 55)
    int_path = f"comp_rel_{idx:04d}_intv.png"
    img2.save(IMG_DIR / int_path)

    pairs.append({
        "id": f"comp_rel_{idx:04d}",
        "category": "compositional",
        "subcategory": "relative_composition",
        "intervention": "replace_color",
        "original_image": orig_path,
        "intervened_image": int_path,
        "question": f"Is there a {c1} {s1} above a {c2} {s2}?",
        "answer_original": "yes",
        "answer_intervened": "no",
        "should_change": True,
    })
    return pairs


def gen_counting(idx):
    """Several shapes; question: how many [color] [shape]s? Intervention: add/remove one."""
    pairs = []
    target_color = random.choice(COLOR_NAMES)
    target_shape = random.choice(SHAPES)
    n_target = random.randint(2, 4)
    n_distractor = random.randint(2, 4)

    def _place_objs(n_t, n_d):
        all_pos = []
        attempts = 0
        while len(all_pos) < n_t + n_d and attempts < 200:
            px = random.randint(80, 432)
            py = random.randint(80, 432)
            if all(abs(px - ox) > 80 or abs(py - oy) > 80 for ox, oy in all_pos):
                all_pos.append((px, py))
            attempts += 1
        return all_pos

    positions = _place_objs(n_target, n_distractor)
    if len(positions) < n_target + n_distractor:
        n_distractor = len(positions) - n_target

    # Draw original
    img = Image.new("RGB", IMG_SIZE, BG_COLOR)
    d = ImageDraw.Draw(img)
    for i in range(n_target):
        draw_shape(d, target_shape, target_color, positions[i][0], positions[i][1], 38)
    for i in range(n_target, n_target + n_distractor):
        d_color = random.choice([c for c in COLOR_NAMES if c != target_color])
        d_shape = random.choice([s for s in SHAPES if s != target_shape])
        draw_shape(d, d_shape, d_color, positions[i][0], positions[i][1], 38)
    orig_path = f"count_{idx:04d}_orig.png"
    img.save(IMG_DIR / orig_path)

    # Intervention: remove one target
    img2 = Image.new("RGB", IMG_SIZE, BG_COLOR)
    d2 = ImageDraw.Draw(img2)
    for i in range(n_target - 1):
        draw_shape(d2, target_shape, target_color, positions[i][0], positions[i][1], 38)
    for i in range(n_target, n_target + n_distractor):
        d_color = random.choice([c for c in COLOR_NAMES if c != target_color])
        d_shape = random.choice([s for s in SHAPES if s != target_shape])
        draw_shape(d2, d_shape, d_color, positions[i][0], positions[i][1], 38)
    int_path = f"count_{idx:04d}_intv.png"
    img2.save(IMG_DIR / int_path)

    pairs.append({
        "id": f"count_{idx:04d}",
        "category": "counting",
        "subcategory": "count_removal",
        "intervention": "remove_object",
        "original_image": orig_path,
        "intervened_image": int_path,
        "question": f"How many {target_color} {target_shape}s are in the scene?",
        "answer_original": str(n_target),
        "answer_intervened": str(n_target - 1),
        "should_change": True,
    })
    return pairs


def gen_counting_nochange(idx):
    """Remove a distractor; count of target should NOT change."""
    pairs = []
    target_color = random.choice(COLOR_NAMES)
    target_shape = random.choice(SHAPES)
    n_target = random.randint(2, 4)

    positions = []
    attempts = 0
    while len(positions) < n_target + 2 and attempts < 200:
        px = random.randint(80, 432)
        py = random.randint(80, 432)
        if all(abs(px - ox) > 80 or abs(py - oy) > 80 for ox, oy in positions):
            positions.append((px, py))
        attempts += 1

    if len(positions) < n_target + 2:
        return []

    d_color = random.choice([c for c in COLOR_NAMES if c != target_color])
    d_shape = random.choice([s for s in SHAPES if s != target_shape])

    # Original
    img = Image.new("RGB", IMG_SIZE, BG_COLOR)
    d = ImageDraw.Draw(img)
    for i in range(n_target):
        draw_shape(d, target_shape, target_color, positions[i][0], positions[i][1], 38)
    for i in range(n_target, len(positions)):
        draw_shape(d, d_shape, d_color, positions[i][0], positions[i][1], 38)
    orig_path = f"count_nc_{idx:04d}_orig.png"
    img.save(IMG_DIR / orig_path)

    # Intervention: remove a distractor (not a target)
    img2 = Image.new("RGB", IMG_SIZE, BG_COLOR)
    d2 = ImageDraw.Draw(img2)
    for i in range(n_target):
        draw_shape(d2, target_shape, target_color, positions[i][0], positions[i][1], 38)
    for i in range(n_target, len(positions) - 1):
        draw_shape(d2, d_shape, d_color, positions[i][0], positions[i][1], 38)
    int_path = f"count_nc_{idx:04d}_intv.png"
    img2.save(IMG_DIR / int_path)

    pairs.append({
        "id": f"count_nc_{idx:04d}",
        "category": "counting",
        "subcategory": "count_no_change",
        "intervention": "remove_distractor",
        "original_image": orig_path,
        "intervened_image": int_path,
        "question": f"How many {target_color} {target_shape}s are in the scene?",
        "answer_original": str(n_target),
        "answer_intervened": str(n_target),
        "should_change": False,
    })
    return pairs


def gen_occlusion(idx):
    """Object partially behind another; question: is object fully visible?"""
    pairs = []
    c_front = random.choice(COLOR_NAMES)
    c_back = random.choice([c for c in COLOR_NAMES if c != c_front])
    s_front = random.choice(SHAPES)
    s_back = random.choice(SHAPES)

    # Original: back object partially hidden
    img = Image.new("RGB", IMG_SIZE, BG_COLOR)
    d = ImageDraw.Draw(img)
    draw_shape(d, s_back, c_back, 230, 256, 55)
    draw_shape(d, s_front, c_front, 290, 256, 60)
    add_label(d, "target", 195, 315)
    orig_path = f"occl_{idx:04d}_orig.png"
    img.save(IMG_DIR / orig_path)

    # Intervention: move front object away
    img2 = Image.new("RGB", IMG_SIZE, BG_COLOR)
    d2 = ImageDraw.Draw(img2)
    draw_shape(d2, s_back, c_back, 230, 256, 55)
    draw_shape(d2, s_front, c_front, 410, 256, 60)
    add_label(d2, "target", 195, 315)
    int_path = f"occl_{idx:04d}_intv.png"
    img2.save(IMG_DIR / int_path)

    pairs.append({
        "id": f"occl_{idx:04d}",
        "category": "occlusion",
        "subcategory": "partial_occlusion",
        "intervention": "remove_occluder",
        "original_image": orig_path,
        "intervened_image": int_path,
        "question": f"Is the target ({c_back} {s_back}) fully visible without any occlusion?",
        "answer_original": "no",
        "answer_intervened": "yes",
        "should_change": True,
    })
    return pairs


def gen_spatial_nochange(idx):
    """Change color of one object; spatial relation should NOT change."""
    pairs = []
    c1, c2 = random.sample(COLOR_NAMES, 2)
    s1, s2 = random.sample(SHAPES, 2)

    img = Image.new("RGB", IMG_SIZE, BG_COLOR)
    d = ImageDraw.Draw(img)
    draw_shape(d, s1, c1, 160, 256, 55)
    draw_shape(d, s2, c2, 352, 256, 55)
    add_label(d, "A", 145, 320)
    add_label(d, "B", 337, 320)
    orig_path = f"spatial_nc_{idx:04d}_orig.png"
    img.save(IMG_DIR / orig_path)

    # Intervention: change color of A (position stays)
    c3 = random.choice([c for c in COLOR_NAMES if c != c1])
    img2 = Image.new("RGB", IMG_SIZE, BG_COLOR)
    d2 = ImageDraw.Draw(img2)
    draw_shape(d2, s1, c3, 160, 256, 55)
    draw_shape(d2, s2, c2, 352, 256, 55)
    add_label(d2, "A", 145, 320)
    add_label(d2, "B", 337, 320)
    int_path = f"spatial_nc_{idx:04d}_intv.png"
    img2.save(IMG_DIR / int_path)

    pairs.append({
        "id": f"spatial_nc_{idx:04d}",
        "category": "spatial",
        "subcategory": "left_right_no_change",
        "intervention": "replace_color",
        "original_image": orig_path,
        "intervened_image": int_path,
        "question": "Is A to the left of B?",
        "answer_original": "yes",
        "answer_intervened": "yes",
        "should_change": False,
    })
    return pairs


# ── Main generation ───────────────────────────────────────────────────────

def generate_all():
    all_pairs = []
    
    generators_change = [
        (gen_spatial_leftright, 40),
        (gen_spatial_above_below, 30),
        (gen_spatial_inside, 20),
        (gen_causal_arrow, 30),
        (gen_causal_spill, 20),
        (gen_causal_chain, 20),
        (gen_compositional_attribute, 35),
        (gen_compositional_relative, 25),
        (gen_counting, 30),
        (gen_occlusion, 25),
    ]
    
    generators_nochange = [
        (gen_counting_nochange, 25),
        (gen_spatial_nochange, 25),
    ]

    idx = 0
    for gen_fn, count in generators_change:
        for i in range(count):
            pairs = gen_fn(idx)
            all_pairs.extend(pairs)
            idx += 1

    for gen_fn, count in generators_nochange:
        for i in range(count):
            pairs = gen_fn(idx)
            all_pairs.extend(pairs)
            idx += 1

    return all_pairs


if __name__ == "__main__":
    print("Generating CounterBench dataset...")
    pairs = generate_all()
    print(f"Generated {len(pairs)} counterfactual pairs")
    
    # Summary statistics
    from collections import Counter
    cats = Counter(p["category"] for p in pairs)
    subcats = Counter(p["subcategory"] for p in pairs)
    changes = Counter(p["should_change"] for p in pairs)
    interventions = Counter(p["intervention"] for p in pairs)
    
    print(f"\nBy category: {dict(cats)}")
    print(f"By subcategory: {dict(subcats)}")
    print(f"Should change: {dict(changes)}")
    print(f"Interventions: {dict(interventions)}")
    
    # Save metadata
    meta_path = OUTPUT_DIR / "counterbench_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(pairs, f, indent=2)
    print(f"\nMetadata saved to {meta_path}")
    print(f"Images saved to {IMG_DIR}")
