"""
CounterBench: Synthetic Scene Generator for Counterfactual Consistency Testing
Generates 500 paired images (original + intervened) across 5 task categories.
"""

import os
import json
import random
import math
from PIL import Image, ImageDraw, ImageFont

random.seed(42)

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "images")
META_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "benchmark.json")
os.makedirs(OUT_DIR, exist_ok=True)

W, H = 512, 512
BG_COLOR = (245, 245, 245)

COLORS = {
    "red": (220, 50, 50),
    "blue": (50, 80, 220),
    "green": (50, 180, 80),
    "yellow": (230, 200, 40),
    "purple": (150, 50, 200),
    "orange": (240, 140, 30),
    "cyan": (40, 200, 210),
    "pink": (230, 100, 170),
}

SHAPES = ["circle", "square", "triangle", "diamond"]


def draw_shape(draw, shape, cx, cy, size, color_rgb, outline=None):
    """Draw a shape centered at (cx, cy) with given size."""
    s = size // 2
    if shape == "circle":
        draw.ellipse([cx - s, cy - s, cx + s, cy + s], fill=color_rgb, outline=outline, width=2)
    elif shape == "square":
        draw.rectangle([cx - s, cy - s, cx + s, cy + s], fill=color_rgb, outline=outline, width=2)
    elif shape == "triangle":
        pts = [(cx, cy - s), (cx - s, cy + s), (cx + s, cy + s)]
        draw.polygon(pts, fill=color_rgb, outline=outline)
    elif shape == "diamond":
        pts = [(cx, cy - s), (cx + s, cy), (cx, cy + s), (cx - s, cy)]
        draw.polygon(pts, fill=color_rgb, outline=outline)


def draw_arrow(draw, x1, y1, x2, y2, color=(60, 60, 60), width=3):
    """Draw an arrow from (x1,y1) to (x2,y2)."""
    draw.line([(x1, y1), (x2, y2)], fill=color, width=width)
    angle = math.atan2(y2 - y1, x2 - x1)
    hl = 15
    a1 = angle + math.pi + 0.4
    a2 = angle + math.pi - 0.4
    draw.line([(x2, y2), (x2 + hl * math.cos(a1), y2 + hl * math.sin(a1))], fill=color, width=width)
    draw.line([(x2, y2), (x2 + hl * math.cos(a2), y2 + hl * math.sin(a2))], fill=color, width=width)


def draw_container(draw, cx, cy, w, h, color_rgb, label=None):
    """Draw a rectangular container (open top or closed)."""
    draw.rectangle([cx - w//2, cy - h//2, cx + w//2, cy + h//2], 
                   fill=None, outline=color_rgb, width=3)


def add_label(draw, text, x, y, color=(40, 40, 40)):
    """Add text label."""
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
    except:
        font = ImageFont.load_default()
    draw.text((x, y), text, fill=color, font=font)


# ============================================================================
# TASK 1: SPATIAL RELATIONS  (100 pairs)
# Q: "Is {shape_a} to the {direction} of {shape_b}?"
# Intervention: swap positions -> answer flips
# ============================================================================
def gen_spatial(idx):
    items = []
    for i in range(100):
        shape_a, shape_b = random.sample(SHAPES, 2)
        color_a_name, color_b_name = random.sample(list(COLORS.keys()), 2)
        color_a, color_b = COLORS[color_a_name], COLORS[color_b_name]
        size_a = random.randint(50, 80)
        size_b = random.randint(50, 80)
        
        # Choose spatial relation
        relation = random.choice(["left", "right", "above", "below"])
        
        # Position A and B based on relation (A is to the {relation} of B)
        margin = 100
        if relation in ["left", "right"]:
            y_a = y_b = random.randint(H//3, 2*H//3)
            if relation == "left":
                x_a = random.randint(margin, W//2 - 40)
                x_b = random.randint(W//2 + 40, W - margin)
            else:
                x_a = random.randint(W//2 + 40, W - margin)
                x_b = random.randint(margin, W//2 - 40)
        else:
            x_a = x_b = random.randint(W//3, 2*W//3)
            if relation == "above":
                y_a = random.randint(margin, H//2 - 40)
                y_b = random.randint(H//2 + 40, H - margin)
            else:
                y_a = random.randint(H//2 + 40, H - margin)
                y_b = random.randint(margin, H//2 - 40)
        
        # Original image
        img_orig = Image.new("RGB", (W, H), BG_COLOR)
        d = ImageDraw.Draw(img_orig)
        draw_shape(d, shape_a, x_a, y_a, size_a, color_a, outline=(30,30,30))
        draw_shape(d, shape_b, x_b, y_b, size_b, color_b, outline=(30,30,30))
        add_label(d, "A", x_a - 8, y_a - size_a//2 - 25)
        add_label(d, "B", x_b - 8, y_b - size_b//2 - 25)
        
        # Intervened: swap A and B positions
        img_int = Image.new("RGB", (W, H), BG_COLOR)
        d2 = ImageDraw.Draw(img_int)
        draw_shape(d2, shape_a, x_b, y_b, size_a, color_a, outline=(30,30,30))
        draw_shape(d2, shape_b, x_a, y_a, size_b, color_b, outline=(30,30,30))
        add_label(d2, "A", x_b - 8, y_b - size_a//2 - 25)
        add_label(d2, "B", x_a - 8, y_a - size_b//2 - 25)
        
        # Save
        id_str = f"spatial_{i:03d}"
        img_orig.save(os.path.join(OUT_DIR, f"{id_str}_orig.png"))
        img_int.save(os.path.join(OUT_DIR, f"{id_str}_int.png"))
        
        question = f"Is the {color_a_name} {shape_a} (labeled A) to the {relation} of the {color_b_name} {shape_b} (labeled B)? Answer yes or no."
        
        items.append({
            "id": id_str,
            "category": "spatial",
            "question": question,
            "original_answer": "yes",
            "intervened_answer": "no",
            "should_flip": True,
            "intervention": f"Swapped positions of A and B",
            "original_image": f"{id_str}_orig.png",
            "intervened_image": f"{id_str}_int.png",
            "details": {
                "shape_a": shape_a, "color_a": color_a_name,
                "shape_b": shape_b, "color_b": color_b_name,
                "relation": relation
            }
        })
    return items


# ============================================================================
# TASK 2: ATTRIBUTE BINDING  (100 pairs)
# Q: "What color is the {shape}?"
# Intervention: change color of that shape -> answer changes
# ============================================================================
def gen_attribute(idx):
    items = []
    for i in range(100):
        n_shapes = random.randint(2, 4)
        shapes_used = random.sample(SHAPES * 2, n_shapes)
        colors_used = random.sample(list(COLORS.keys()), n_shapes)
        
        positions = []
        for _ in range(n_shapes):
            while True:
                x = random.randint(100, W - 100)
                y = random.randint(100, H - 100)
                ok = all(abs(x - px) > 90 or abs(y - py) > 90 for px, py in positions)
                if ok:
                    positions.append((x, y))
                    break
        
        sizes = [random.randint(50, 75) for _ in range(n_shapes)]
        
        # Target shape to ask about
        target_idx = random.randint(0, n_shapes - 1)
        target_shape = shapes_used[target_idx]
        target_color = colors_used[target_idx]
        
        # New color for intervention
        remaining_colors = [c for c in COLORS.keys() if c != target_color]
        new_color = random.choice(remaining_colors)
        
        # Original
        img_orig = Image.new("RGB", (W, H), BG_COLOR)
        d = ImageDraw.Draw(img_orig)
        for j in range(n_shapes):
            draw_shape(d, shapes_used[j], positions[j][0], positions[j][1],
                       sizes[j], COLORS[colors_used[j]], outline=(30,30,30))
        
        # Intervened: change color of target
        img_int = Image.new("RGB", (W, H), BG_COLOR)
        d2 = ImageDraw.Draw(img_int)
        for j in range(n_shapes):
            c = new_color if j == target_idx else colors_used[j]
            draw_shape(d2, shapes_used[j], positions[j][0], positions[j][1],
                       sizes[j], COLORS[c], outline=(30,30,30))
        
        id_str = f"attribute_{i:03d}"
        img_orig.save(os.path.join(OUT_DIR, f"{id_str}_orig.png"))
        img_int.save(os.path.join(OUT_DIR, f"{id_str}_int.png"))
        
        # Describe position for uniqueness
        pos_desc = ""
        tx, ty = positions[target_idx]
        if tx < W//3:
            pos_desc = "leftmost"
        elif tx > 2*W//3:
            pos_desc = "rightmost"
        elif ty < H//3:
            pos_desc = "topmost"
        else:
            pos_desc = "central"
        
        question = f"What color is the {pos_desc} {target_shape}? Answer with a single color word."
        
        items.append({
            "id": id_str,
            "category": "attribute",
            "question": question,
            "original_answer": target_color,
            "intervened_answer": new_color,
            "should_flip": True,
            "intervention": f"Changed color of {target_shape} from {target_color} to {new_color}",
            "original_image": f"{id_str}_orig.png",
            "intervened_image": f"{id_str}_int.png",
            "details": {
                "target_shape": target_shape,
                "original_color": target_color,
                "new_color": new_color,
                "n_shapes": n_shapes
            }
        })
    return items


# ============================================================================
# TASK 3: COUNTING  (100 pairs)
# Q: "How many {color} {shape}s are there?"
# Intervention: add or remove one -> count changes
# ============================================================================
def gen_counting(idx):
    items = []
    for i in range(100):
        target_shape = random.choice(SHAPES)
        target_color_name = random.choice(list(COLORS.keys()))
        target_color = COLORS[target_color_name]
        
        n_target = random.randint(2, 6)
        n_distractor = random.randint(1, 3)
        
        # Generate positions for targets
        all_positions = []
        for _ in range(n_target + n_distractor + 2):
            for attempt in range(50):
                x = random.randint(80, W - 80)
                y = random.randint(80, H - 80)
                ok = all(abs(x-px) > 70 or abs(y-py) > 70 for px, py in all_positions)
                if ok:
                    all_positions.append((x, y))
                    break
        
        target_positions = all_positions[:n_target]
        distractor_positions = all_positions[n_target:n_target + n_distractor]
        extra_position = all_positions[n_target + n_distractor] if len(all_positions) > n_target + n_distractor else (W//2, H//2)
        
        # Distractor shapes/colors
        dist_shapes = [random.choice([s for s in SHAPES if s != target_shape]) for _ in range(n_distractor)]
        dist_colors = [COLORS[random.choice([c for c in COLORS.keys() if c != target_color_name])] for _ in range(n_distractor)]
        
        size = random.randint(40, 60)
        
        # Decide intervention: add or remove
        if n_target >= 3:
            action = random.choice(["add", "remove"])
        else:
            action = "add"
        
        # Original
        img_orig = Image.new("RGB", (W, H), BG_COLOR)
        d = ImageDraw.Draw(img_orig)
        for pos in target_positions:
            draw_shape(d, target_shape, pos[0], pos[1], size, target_color, outline=(30,30,30))
        for j, pos in enumerate(distractor_positions):
            draw_shape(d, dist_shapes[j], pos[0], pos[1], size, dist_colors[j], outline=(30,30,30))
        
        # Intervened
        img_int = Image.new("RGB", (W, H), BG_COLOR)
        d2 = ImageDraw.Draw(img_int)
        
        if action == "add":
            new_count = n_target + 1
            for pos in target_positions:
                draw_shape(d2, target_shape, pos[0], pos[1], size, target_color, outline=(30,30,30))
            draw_shape(d2, target_shape, extra_position[0], extra_position[1], size, target_color, outline=(30,30,30))
        else:
            new_count = n_target - 1
            for pos in target_positions[:-1]:
                draw_shape(d2, target_shape, pos[0], pos[1], size, target_color, outline=(30,30,30))
        
        for j, pos in enumerate(distractor_positions):
            draw_shape(d2, dist_shapes[j], pos[0], pos[1], size, dist_colors[j], outline=(30,30,30))
        
        id_str = f"counting_{i:03d}"
        img_orig.save(os.path.join(OUT_DIR, f"{id_str}_orig.png"))
        img_int.save(os.path.join(OUT_DIR, f"{id_str}_int.png"))
        
        question = f"How many {target_color_name} {target_shape}s are in the image? Answer with a single number."
        
        items.append({
            "id": id_str,
            "category": "counting",
            "question": question,
            "original_answer": str(n_target),
            "intervened_answer": str(new_count),
            "should_flip": True,
            "intervention": f"{'Added' if action == 'add' else 'Removed'} one {target_color_name} {target_shape}",
            "original_image": f"{id_str}_orig.png",
            "intervened_image": f"{id_str}_int.png",
            "details": {
                "target_shape": target_shape,
                "target_color": target_color_name,
                "original_count": n_target,
                "new_count": new_count,
                "action": action
            }
        })
    return items


# ============================================================================
# TASK 4: CONTAINMENT  (100 pairs)
# Q: "Is the {shape} inside the container?"
# Intervention: move shape in/out -> answer flips
# ============================================================================
def gen_containment(idx):
    items = []
    for i in range(100):
        shape = random.choice(SHAPES)
        color_name = random.choice(list(COLORS.keys()))
        color = COLORS[color_name]
        container_color = COLORS[random.choice([c for c in COLORS.keys() if c != color_name])]
        
        # Container position and size
        cx, cy = W//2 + random.randint(-60, 60), H//2 + random.randint(-40, 40)
        cw, ch = random.randint(140, 200), random.randint(140, 200)
        shape_size = random.randint(35, 55)
        
        # Inside position
        inside_x = cx + random.randint(-cw//4, cw//4)
        inside_y = cy + random.randint(-ch//4, ch//4)
        
        # Outside position
        side = random.choice(["left", "right", "above", "below"])
        if side == "left":
            outside_x = cx - cw//2 - shape_size - random.randint(20, 50)
            outside_y = cy + random.randint(-30, 30)
        elif side == "right":
            outside_x = cx + cw//2 + shape_size + random.randint(20, 50)
            outside_y = cy + random.randint(-30, 30)
        elif side == "above":
            outside_x = cx + random.randint(-30, 30)
            outside_y = cy - ch//2 - shape_size - random.randint(20, 50)
        else:
            outside_x = cx + random.randint(-30, 30)
            outside_y = cy + ch//2 + shape_size + random.randint(20, 50)
        
        # Clamp
        outside_x = max(60, min(W-60, outside_x))
        outside_y = max(60, min(H-60, outside_y))
        
        originally_inside = random.choice([True, False])
        
        # Original
        img_orig = Image.new("RGB", (W, H), BG_COLOR)
        d = ImageDraw.Draw(img_orig)
        draw_container(d, cx, cy, cw, ch, container_color)
        if originally_inside:
            draw_shape(d, shape, inside_x, inside_y, shape_size, color, outline=(30,30,30))
        else:
            draw_shape(d, shape, outside_x, outside_y, shape_size, color, outline=(30,30,30))
        
        # Intervened: flip inside/outside
        img_int = Image.new("RGB", (W, H), BG_COLOR)
        d2 = ImageDraw.Draw(img_int)
        draw_container(d2, cx, cy, cw, ch, container_color)
        if originally_inside:
            draw_shape(d2, shape, outside_x, outside_y, shape_size, color, outline=(30,30,30))
        else:
            draw_shape(d2, shape, inside_x, inside_y, shape_size, color, outline=(30,30,30))
        
        id_str = f"containment_{i:03d}"
        img_orig.save(os.path.join(OUT_DIR, f"{id_str}_orig.png"))
        img_int.save(os.path.join(OUT_DIR, f"{id_str}_int.png"))
        
        question = f"Is the {color_name} {shape} inside the rectangular container? Answer yes or no."
        
        items.append({
            "id": id_str,
            "category": "containment",
            "question": question,
            "original_answer": "yes" if originally_inside else "no",
            "intervened_answer": "no" if originally_inside else "yes",
            "should_flip": True,
            "intervention": f"Moved {shape} {'outside' if originally_inside else 'inside'} the container",
            "original_image": f"{id_str}_orig.png",
            "intervened_image": f"{id_str}_int.png",
            "details": {
                "shape": shape,
                "color": color_name,
                "originally_inside": originally_inside
            }
        })
    return items


# ============================================================================
# TASK 5: CAUSAL / COMPOSITIONAL  (100 pairs)
# Sub-tasks: arrow direction, occlusion, compositional ("red square above blue circle")
# ============================================================================
def gen_causal(idx):
    items = []
    
    # 5a: Arrow pointing (34 items)
    for i in range(34):
        n_shapes = 3
        shape_names = random.sample(SHAPES, n_shapes)
        color_names = random.sample(list(COLORS.keys()), n_shapes)
        
        positions = []
        for _ in range(n_shapes):
            for attempt in range(50):
                x = random.randint(100, W - 100)
                y = random.randint(100, H - 100)
                if all(abs(x-px) > 110 or abs(y-py) > 110 for px, py in positions):
                    positions.append((x, y))
                    break
        
        if len(positions) < n_shapes:
            positions = [(150, 256), (350, 150), (350, 370)][:n_shapes]
        
        target_idx = random.randint(0, n_shapes - 1)
        new_target_idx = (target_idx + 1) % n_shapes
        
        # Arrow starts from center
        arrow_start = (W//2, H - 60)
        
        # Original: arrow points to target
        img_orig = Image.new("RGB", (W, H), BG_COLOR)
        d = ImageDraw.Draw(img_orig)
        for j in range(n_shapes):
            draw_shape(d, shape_names[j], positions[j][0], positions[j][1],
                       60, COLORS[color_names[j]], outline=(30,30,30))
        draw_arrow(d, arrow_start[0], arrow_start[1], 
                   positions[target_idx][0], positions[target_idx][1] + 35)
        
        # Intervened: arrow points to different target
        img_int = Image.new("RGB", (W, H), BG_COLOR)
        d2 = ImageDraw.Draw(img_int)
        for j in range(n_shapes):
            draw_shape(d2, shape_names[j], positions[j][0], positions[j][1],
                       60, COLORS[color_names[j]], outline=(30,30,30))
        draw_arrow(d2, arrow_start[0], arrow_start[1],
                   positions[new_target_idx][0], positions[new_target_idx][1] + 35)
        
        id_str = f"causal_arrow_{i:03d}"
        img_orig.save(os.path.join(OUT_DIR, f"{id_str}_orig.png"))
        img_int.save(os.path.join(OUT_DIR, f"{id_str}_int.png"))
        
        question = f"Which shape does the arrow point to? Answer with the color and shape name (e.g., 'red circle')."
        
        items.append({
            "id": id_str,
            "category": "causal",
            "subcategory": "arrow",
            "question": question,
            "original_answer": f"{color_names[target_idx]} {shape_names[target_idx]}",
            "intervened_answer": f"{color_names[new_target_idx]} {shape_names[new_target_idx]}",
            "should_flip": True,
            "intervention": f"Redirected arrow from {color_names[target_idx]} {shape_names[target_idx]} to {color_names[new_target_idx]} {shape_names[new_target_idx]}",
            "original_image": f"{id_str}_orig.png",
            "intervened_image": f"{id_str}_int.png",
        })
    
    # 5b: Occlusion (33 items)
    for i in range(33):
        shape_front = random.choice(SHAPES)
        shape_back = random.choice(SHAPES)
        color_front_name = random.choice(list(COLORS.keys()))
        color_back_name = random.choice([c for c in COLORS.keys() if c != color_front_name])
        
        cx, cy = W//2, H//2
        offset = random.randint(25, 45)
        size_front = random.randint(70, 90)
        size_back = random.randint(70, 90)
        
        # Original: front occludes back
        img_orig = Image.new("RGB", (W, H), BG_COLOR)
        d = ImageDraw.Draw(img_orig)
        draw_shape(d, shape_back, cx - offset, cy, size_back, COLORS[color_back_name], outline=(30,30,30))
        draw_shape(d, shape_front, cx + offset, cy, size_front, COLORS[color_front_name], outline=(30,30,30))
        
        # Intervened: remove front shape -> back fully visible
        img_int = Image.new("RGB", (W, H), BG_COLOR)
        d2 = ImageDraw.Draw(img_int)
        draw_shape(d2, shape_back, cx - offset, cy, size_back, COLORS[color_back_name], outline=(30,30,30))
        
        id_str = f"causal_occlude_{i:03d}"
        img_orig.save(os.path.join(OUT_DIR, f"{id_str}_orig.png"))
        img_int.save(os.path.join(OUT_DIR, f"{id_str}_int.png"))
        
        question = f"How many shapes are visible in the image? Answer with a single number."
        
        items.append({
            "id": id_str,
            "category": "causal",
            "subcategory": "occlusion",
            "question": question,
            "original_answer": "2",
            "intervened_answer": "1",
            "should_flip": True,
            "intervention": f"Removed the {color_front_name} {shape_front} (front shape)",
            "original_image": f"{id_str}_orig.png",
            "intervened_image": f"{id_str}_int.png",
        })
    
    # 5c: Compositional description (33 items)  
    # Q: "Is there a red square above a blue circle?" 
    # Intervention: rearrange so description no longer holds
    for i in range(33):
        shape_top = random.choice(SHAPES)
        shape_bottom = random.choice(SHAPES)
        color_top_name, color_bottom_name = random.sample(list(COLORS.keys()), 2)
        
        # Original: composition holds (top above bottom)
        x_pos = W//2 + random.randint(-80, 80)
        y_top = H//3 + random.randint(-30, 30)
        y_bottom = 2*H//3 + random.randint(-30, 30)
        size = random.randint(55, 75)
        
        img_orig = Image.new("RGB", (W, H), BG_COLOR)
        d = ImageDraw.Draw(img_orig)
        draw_shape(d, shape_top, x_pos, y_top, size, COLORS[color_top_name], outline=(30,30,30))
        draw_shape(d, shape_bottom, x_pos, y_bottom, size, COLORS[color_bottom_name], outline=(30,30,30))
        
        # Intervened: swap vertical positions
        img_int = Image.new("RGB", (W, H), BG_COLOR)
        d2 = ImageDraw.Draw(img_int)
        draw_shape(d2, shape_top, x_pos, y_bottom, size, COLORS[color_top_name], outline=(30,30,30))
        draw_shape(d2, shape_bottom, x_pos, y_top, size, COLORS[color_bottom_name], outline=(30,30,30))
        
        id_str = f"causal_comp_{i:03d}"
        img_orig.save(os.path.join(OUT_DIR, f"{id_str}_orig.png"))
        img_int.save(os.path.join(OUT_DIR, f"{id_str}_int.png"))
        
        question = f"Is the {color_top_name} {shape_top} above the {color_bottom_name} {shape_bottom}? Answer yes or no."
        
        items.append({
            "id": id_str,
            "category": "causal",
            "subcategory": "compositional",
            "question": question,
            "original_answer": "yes",
            "intervened_answer": "no",
            "should_flip": True,
            "intervention": f"Swapped vertical positions of the two shapes",
            "original_image": f"{id_str}_orig.png",
            "intervened_image": f"{id_str}_int.png",
        })
    
    return items


# ============================================================================
# NEGATIVE CONTROLS: Irrelevant interventions (should NOT flip)
# Change background shade, add unrelated distractor far away, etc.
# ============================================================================
def gen_negative_controls():
    """Generate 50 pairs where intervention is irrelevant - answer should NOT change."""
    items = []
    for i in range(50):
        shape = random.choice(SHAPES)
        color_name = random.choice(list(COLORS.keys()))
        color = COLORS[color_name]
        
        x, y = W//2 + random.randint(-80, 80), H//2 + random.randint(-60, 60)
        size = random.randint(55, 80)
        
        # Original
        img_orig = Image.new("RGB", (W, H), BG_COLOR)
        d = ImageDraw.Draw(img_orig)
        draw_shape(d, shape, x, y, size, color, outline=(30,30,30))
        
        # Intervention: add unrelated shape in corner (doesn't affect answer about target)
        dist_shape = random.choice([s for s in SHAPES if s != shape])
        dist_color_name = random.choice([c for c in COLORS.keys() if c != color_name])
        corner_x = random.choice([60, W-60])
        corner_y = random.choice([60, H-60])
        
        img_int = Image.new("RGB", (W, H), BG_COLOR)
        d2 = ImageDraw.Draw(img_int)
        draw_shape(d2, shape, x, y, size, color, outline=(30,30,30))
        draw_shape(d2, dist_shape, corner_x, corner_y, 35, COLORS[dist_color_name], outline=(30,30,30))
        
        id_str = f"negctrl_{i:03d}"
        img_orig.save(os.path.join(OUT_DIR, f"{id_str}_orig.png"))
        img_int.save(os.path.join(OUT_DIR, f"{id_str}_int.png"))
        
        question = f"What color is the {shape} in the center of the image? Answer with a single color word."
        
        items.append({
            "id": id_str,
            "category": "negative_control",
            "question": question,
            "original_answer": color_name,
            "intervened_answer": color_name,  # Same - should NOT flip
            "should_flip": False,
            "intervention": f"Added unrelated {dist_color_name} {dist_shape} in corner",
            "original_image": f"{id_str}_orig.png",
            "intervened_image": f"{id_str}_int.png",
        })
    return items


def main():
    print("Generating CounterBench scenes...")
    all_items = []
    
    print("  [1/6] Spatial relations (100 pairs)...")
    all_items.extend(gen_spatial(0))
    
    print("  [2/6] Attribute binding (100 pairs)...")
    all_items.extend(gen_attribute(100))
    
    print("  [3/6] Counting (100 pairs)...")
    all_items.extend(gen_counting(200))
    
    print("  [4/6] Containment (100 pairs)...")
    all_items.extend(gen_containment(300))
    
    print("  [5/6] Causal/compositional (100 pairs)...")
    all_items.extend(gen_causal(400))
    
    print("  [6/6] Negative controls (50 pairs)...")
    all_items.extend(gen_negative_controls())
    
    # Save metadata
    benchmark = {
        "name": "CounterBench",
        "version": "1.0",
        "total_pairs": len(all_items),
        "categories": {
            "spatial": 100,
            "attribute": 100,
            "counting": 100,
            "containment": 100,
            "causal": 100,
            "negative_control": 50
        },
        "items": all_items
    }
    
    with open(META_PATH, "w") as f:
        json.dump(benchmark, f, indent=2)
    
    print(f"\nDone! Generated {len(all_items)} pairs.")
    print(f"Images saved to: {OUT_DIR}")
    print(f"Metadata saved to: {META_PATH}")
    
    # Stats
    flips = sum(1 for item in all_items if item["should_flip"])
    no_flips = sum(1 for item in all_items if not item["should_flip"])
    print(f"Should-flip pairs: {flips}, Should-NOT-flip pairs: {no_flips}")


if __name__ == "__main__":
    main()
