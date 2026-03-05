#!/usr/bin/env python3
"""
UI-Drift Benchmark Generator
Generates 50 base UI screenshots and 5 drift variants each using PIL.
Each page has associated grounded QA pairs with bounding-box evidence.
"""

import os
import json
import random
import hashlib
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np

random.seed(42)
np.random.seed(42)

OUT_DIR = Path(__file__).parent.parent / "data"
IMG_DIR = OUT_DIR / "images"
META_FILE = OUT_DIR / "benchmark.json"

W, H = 1280, 900  # Base resolution

# ── Color themes ──────────────────────────────────────────────
THEMES = {
    "light": {
        "bg": (255, 255, 255), "card": (245, 245, 250), "text": (30, 30, 30),
        "accent": (56, 120, 200), "border": (210, 210, 215),
        "sidebar_bg": (240, 242, 248), "header_bg": (56, 120, 200),
        "header_text": (255, 255, 255), "muted": (130, 130, 140),
        "success": (46, 160, 67), "warning": (210, 150, 30), "danger": (210, 50, 50),
    },
    "dark": {
        "bg": (28, 28, 36), "card": (38, 40, 52), "text": (225, 225, 230),
        "accent": (80, 150, 240), "border": (55, 58, 70),
        "sidebar_bg": (22, 22, 30), "header_bg": (38, 40, 52),
        "header_text": (225, 225, 230), "muted": (140, 140, 155),
        "success": (60, 180, 80), "warning": (230, 170, 50), "danger": (230, 70, 70),
    },
    "sepia": {
        "bg": (245, 235, 220), "card": (238, 225, 205), "text": (60, 45, 30),
        "accent": (150, 90, 40), "border": (200, 185, 165),
        "sidebar_bg": (235, 220, 200), "header_bg": (150, 90, 40),
        "header_text": (255, 245, 230), "muted": (130, 110, 90),
        "success": (80, 140, 60), "warning": (190, 140, 40), "danger": (180, 60, 40),
    },
    "blue": {
        "bg": (230, 240, 255), "card": (220, 232, 250), "text": (20, 30, 60),
        "accent": (30, 80, 180), "border": (180, 200, 230),
        "sidebar_bg": (210, 225, 248), "header_bg": (30, 80, 180),
        "header_text": (255, 255, 255), "muted": (100, 115, 145),
        "success": (40, 150, 80), "warning": (200, 150, 30), "danger": (200, 50, 50),
    },
}

# ── Realistic data for UI content ─────────────────────────────
DASHBOARD_METRICS = [
    ("Total Revenue", "$1,284,560", "+12.4%"),
    ("Active Users", "48,291", "+8.7%"),
    ("Conversion Rate", "3.42%", "-0.3%"),
    ("Avg Session Time", "4m 32s", "+1.1%"),
    ("Bounce Rate", "34.8%", "-2.1%"),
    ("Page Views", "892,104", "+15.3%"),
    ("New Signups", "2,847", "+22.1%"),
    ("Support Tickets", "142", "-18.4%"),
    ("Server Uptime", "99.97%", "+0.02%"),
    ("API Calls", "12.4M", "+31.2%"),
    ("Error Rate", "0.08%", "-0.03%"),
    ("Load Time", "1.2s", "-0.4s"),
]

TABLE_DATA = [
    ["Invoice #", "Client", "Amount", "Status", "Date"],
    ["INV-0042", "Acme Corp", "$12,450", "Paid", "2025-11-15"],
    ["INV-0043", "TechStart Inc", "$8,200", "Pending", "2025-11-18"],
    ["INV-0044", "Global Media", "$24,800", "Overdue", "2025-10-30"],
    ["INV-0045", "DataFlow LLC", "$5,600", "Paid", "2025-11-20"],
    ["INV-0046", "CloudNet", "$18,300", "Pending", "2025-11-22"],
    ["INV-0047", "PixelWorks", "$3,200", "Paid", "2025-11-25"],
    ["INV-0048", "NexGen AI", "$45,000", "Paid", "2025-11-28"],
]

SETTINGS_ITEMS = [
    ("Account", "Email: alex.chen@company.com"),
    ("Display Name", "Alex Chen"),
    ("Language", "English (US)"),
    ("Timezone", "UTC-8 (Pacific)"),
    ("Two-Factor Auth", "Enabled"),
    ("Notifications", "Email + Push"),
    ("API Key", "sk-...a8f3"),
    ("Plan", "Professional"),
]

ARTICLE_TITLES = [
    "Understanding Transformer Attention Patterns in Vision Models",
    "Best Practices for Deploying ML Models at Scale",
    "A Guide to Efficient Fine-Tuning of Large Language Models",
    "Exploring Multimodal Retrieval for Enterprise Search",
    "Real-Time Object Detection: From YOLO to Modern Approaches",
]

NAV_ITEMS = ["Dashboard", "Analytics", "Users", "Settings", "Reports", "Billing"]

# ── Font helper ────────────────────────────────────────────────
def get_font(size, bold=False):
    """Get a font - try system fonts, fall back to default."""
    font_paths = [
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/SFNSText.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Arial.ttf",
    ]
    if bold:
        font_paths = [
            "/System/Library/Fonts/Helvetica.ttc",
            "/System/Library/Fonts/SFNSText-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        ] + font_paths
    
    for fp in font_paths:
        if os.path.exists(fp):
            try:
                return ImageFont.truetype(fp, size)
            except:
                continue
    return ImageFont.load_default()


# ── Drawing primitives ─────────────────────────────────────────
def draw_rounded_rect(draw, xy, fill, radius=8, outline=None):
    x0, y0, x1, y1 = xy
    draw.rectangle([x0+radius, y0, x1-radius, y1], fill=fill, outline=outline)
    draw.rectangle([x0, y0+radius, x1, y1-radius], fill=fill, outline=outline)
    draw.pieslice([x0, y0, x0+2*radius, y0+2*radius], 180, 270, fill=fill)
    draw.pieslice([x1-2*radius, y0, x1, y0+2*radius], 270, 360, fill=fill)
    draw.pieslice([x0, y1-2*radius, x0+2*radius, y1], 90, 180, fill=fill)
    draw.pieslice([x1-2*radius, y1-2*radius, x1, y1], 0, 90, fill=fill)


def draw_sidebar(draw, theme, items, width=220, active_idx=0):
    """Draw sidebar navigation. Returns bounding boxes for items."""
    t = THEMES[theme]
    draw.rectangle([0, 0, width, H], fill=t["sidebar_bg"])
    draw.line([width, 0, width, H], fill=t["border"], width=1)
    
    # Logo area
    font_logo = get_font(18, bold=True)
    draw.text((20, 25), "AppName", fill=t["accent"], font=font_logo)
    
    bboxes = {}
    font_nav = get_font(14)
    y = 80
    for i, item in enumerate(items):
        if i == active_idx:
            draw_rounded_rect(draw, (10, y-4, width-10, y+28), fill=t["accent"], radius=6)
            draw.text((24, y), item, fill=(255,255,255), font=font_nav)
        else:
            draw.text((24, y), item, fill=t["text"], font=font_nav)
        bboxes[item] = (10, y-4, width-10, y+28)
        y += 40
    return bboxes, width


def draw_header(draw, theme, title, x_offset=0):
    """Draw a top header bar."""
    t = THEMES[theme]
    draw.rectangle([x_offset, 0, W, 56], fill=t["header_bg"])
    font_title = get_font(16, bold=True)
    draw.text((x_offset + 20, 18), title, fill=t["header_text"], font=font_title)
    
    # Search bar
    sx = W - 280
    draw_rounded_rect(draw, (sx, 14, sx+200, 42), fill=t["card"], radius=6)
    font_small = get_font(12)
    draw.text((sx+10, 20), "Search...", fill=t["muted"], font=font_small)
    return {"header_title": (x_offset+20, 18, x_offset+300, 42)}


# ── Page generators ────────────────────────────────────────────
def gen_dashboard(theme="light", sidebar=True, page_id=0):
    """Generate a dashboard page with metric cards and a chart area."""
    t = THEMES[theme]
    img = Image.new("RGB", (W, H), t["bg"])
    draw = ImageDraw.Draw(img)
    
    bboxes = {}
    x_off = 0
    
    if sidebar:
        sb_bboxes, x_off = draw_sidebar(draw, theme, NAV_ITEMS, active_idx=0)
        bboxes.update(sb_bboxes)
    
    hdr_bboxes = draw_header(draw, theme, "Dashboard Overview", x_off)
    bboxes.update(hdr_bboxes)
    
    # Metric cards (2 rows x 4 cols)
    metrics = DASHBOARD_METRICS[page_id*4:(page_id+1)*4]
    if len(metrics) < 4:
        metrics = DASHBOARD_METRICS[:4]
    
    font_label = get_font(11)
    font_value = get_font(22, bold=True)
    font_change = get_font(11)
    
    card_w = (W - x_off - 80) // 4
    for i, (label, value, change) in enumerate(metrics):
        cx = x_off + 20 + i * (card_w + 12)
        cy = 76
        draw_rounded_rect(draw, (cx, cy, cx+card_w, cy+100), fill=t["card"], radius=8,
                         outline=t["border"])
        draw.text((cx+16, cy+14), label, fill=t["muted"], font=font_label)
        draw.text((cx+16, cy+38), value, fill=t["text"], font=font_value)
        
        chg_color = t["success"] if change.startswith("+") else t["danger"]
        draw.text((cx+16, cy+72), change, fill=chg_color, font=font_change)
        bboxes[f"metric_{label}"] = (cx, cy, cx+card_w, cy+100)
    
    # Chart area (simulated bar chart)
    chart_x = x_off + 20
    chart_y = 200
    chart_w = W - x_off - 40
    chart_h = 300
    draw_rounded_rect(draw, (chart_x, chart_y, chart_x+chart_w, chart_y+chart_h),
                     fill=t["card"], radius=8, outline=t["border"])
    draw.text((chart_x+16, chart_y+12), "Monthly Revenue", fill=t["text"], font=get_font(14, True))
    
    # Draw bars
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    bar_values = [65, 72, 58, 80, 92, 85, 78, 95, 88, 102, 98, 115]
    max_val = max(bar_values)
    bar_w = (chart_w - 80) // 12
    font_tiny = get_font(9)
    
    for i, (m, v) in enumerate(zip(months, bar_values)):
        bx = chart_x + 40 + i * bar_w
        bar_h = int((v / max_val) * 200)
        by = chart_y + chart_h - 40 - bar_h
        draw_rounded_rect(draw, (bx+4, by, bx+bar_w-4, chart_y+chart_h-40),
                         fill=t["accent"], radius=4)
        draw.text((bx+8, chart_y+chart_h-32), m, fill=t["muted"], font=font_tiny)
    
    bboxes["chart_area"] = (chart_x, chart_y, chart_x+chart_w, chart_y+chart_h)
    
    # Recent activity table
    tbl_y = chart_y + chart_h + 20
    draw_rounded_rect(draw, (chart_x, tbl_y, chart_x+chart_w, tbl_y+250),
                     fill=t["card"], radius=8, outline=t["border"])
    draw.text((chart_x+16, tbl_y+12), "Recent Transactions", fill=t["text"], font=get_font(14, True))
    
    font_tbl = get_font(11)
    for j, row in enumerate(TABLE_DATA[:6]):
        ry = tbl_y + 42 + j * 32
        if j == 0:
            for k, cell in enumerate(row):
                draw.text((chart_x + 16 + k*180, ry), cell, fill=t["muted"], font=get_font(10, True))
        else:
            for k, cell in enumerate(row):
                color = t["text"]
                if cell == "Paid": color = t["success"]
                elif cell == "Overdue": color = t["danger"]
                elif cell == "Pending": color = t["warning"]
                draw.text((chart_x + 16 + k*180, ry), cell, fill=color, font=font_tbl)
            bboxes[f"row_{row[0]}"] = (chart_x, ry-4, chart_x+chart_w, ry+24)
    
    return img, bboxes


def gen_settings(theme="light", sidebar=True, page_id=0):
    """Generate a settings page."""
    t = THEMES[theme]
    img = Image.new("RGB", (W, H), t["bg"])
    draw = ImageDraw.Draw(img)
    
    bboxes = {}
    x_off = 0
    if sidebar:
        sb_bboxes, x_off = draw_sidebar(draw, theme, NAV_ITEMS, active_idx=3)
        bboxes.update(sb_bboxes)
    
    hdr_bboxes = draw_header(draw, theme, "Account Settings", x_off)
    bboxes.update(hdr_bboxes)
    
    # Settings form
    font_label = get_font(12, bold=True)
    font_value = get_font(13)
    
    form_x = x_off + 40
    form_w = min(600, W - x_off - 80)
    y = 80
    
    draw_rounded_rect(draw, (form_x, y, form_x+form_w, y+len(SETTINGS_ITEMS)*64+30),
                     fill=t["card"], radius=10, outline=t["border"])
    draw.text((form_x+20, y+12), "Profile Settings", fill=t["text"], font=get_font(16, True))
    y += 50
    
    for label, value in SETTINGS_ITEMS:
        draw.text((form_x+20, y), label, fill=t["muted"], font=font_label)
        # Input field
        draw_rounded_rect(draw, (form_x+20, y+20, form_x+form_w-20, y+48),
                         fill=t["bg"], radius=6, outline=t["border"])
        draw.text((form_x+30, y+26), value, fill=t["text"], font=font_value)
        bboxes[f"setting_{label}"] = (form_x+20, y, form_x+form_w-20, y+48)
        y += 64
    
    # Save button
    btn_y = y + 10
    draw_rounded_rect(draw, (form_x+20, btn_y, form_x+160, btn_y+38),
                     fill=t["accent"], radius=6)
    draw.text((form_x+50, btn_y+10), "Save Changes", fill=(255,255,255), font=get_font(13, True))
    bboxes["save_button"] = (form_x+20, btn_y, form_x+160, btn_y+38)
    
    return img, bboxes


def gen_table_page(theme="light", sidebar=True, page_id=0):
    """Generate a data table page."""
    t = THEMES[theme]
    img = Image.new("RGB", (W, H), t["bg"])
    draw = ImageDraw.Draw(img)
    
    bboxes = {}
    x_off = 0
    if sidebar:
        sb_bboxes, x_off = draw_sidebar(draw, theme, NAV_ITEMS, active_idx=4)
        bboxes.update(sb_bboxes)
    
    hdr_bboxes = draw_header(draw, theme, "Reports & Invoices", x_off)
    bboxes.update(hdr_bboxes)
    
    # Filter bar
    fy = 70
    draw_rounded_rect(draw, (x_off+20, fy, W-20, fy+44), fill=t["card"], radius=8, outline=t["border"])
    font_sm = get_font(11)
    draw.text((x_off+36, fy+14), "Filter: All | Paid | Pending | Overdue", fill=t["muted"], font=font_sm)
    bboxes["filter_bar"] = (x_off+20, fy, W-20, fy+44)
    
    # Full table
    tbl_x = x_off + 20
    tbl_y = fy + 60
    tbl_w = W - x_off - 40
    font_hdr = get_font(11, bold=True)
    font_cell = get_font(12)
    
    col_widths = [tbl_w//5]*5
    
    # Header row
    draw.rectangle([tbl_x, tbl_y, tbl_x+tbl_w, tbl_y+36], fill=t["card"])
    for k, cell in enumerate(TABLE_DATA[0]):
        cx = tbl_x + sum(col_widths[:k]) + 12
        draw.text((cx, tbl_y+10), cell, fill=t["muted"], font=font_hdr)
    
    # Data rows
    for j, row in enumerate(TABLE_DATA[1:]):
        ry = tbl_y + 36 + j * 40
        bg = t["bg"] if j % 2 == 0 else t["card"]
        draw.rectangle([tbl_x, ry, tbl_x+tbl_w, ry+40], fill=bg)
        draw.line([tbl_x, ry, tbl_x+tbl_w, ry], fill=t["border"])
        for k, cell in enumerate(row):
            cx = tbl_x + sum(col_widths[:k]) + 12
            color = t["text"]
            if cell == "Paid": color = t["success"]
            elif cell == "Overdue": color = t["danger"]
            elif cell == "Pending": color = t["warning"]
            draw.text((cx, ry+12), cell, fill=color, font=font_cell)
        bboxes[f"tablerow_{row[0]}"] = (tbl_x, ry, tbl_x+tbl_w, ry+40)
    
    return img, bboxes


def gen_article(theme="light", sidebar=True, page_id=0):
    """Generate an article/blog page."""
    t = THEMES[theme]
    img = Image.new("RGB", (W, H), t["bg"])
    draw = ImageDraw.Draw(img)
    
    bboxes = {}
    x_off = 0
    if sidebar:
        sb_bboxes, x_off = draw_sidebar(draw, theme, NAV_ITEMS, active_idx=1)
        bboxes.update(sb_bboxes)
    
    hdr_bboxes = draw_header(draw, theme, "Knowledge Base", x_off)
    bboxes.update(hdr_bboxes)
    
    title = ARTICLE_TITLES[page_id % len(ARTICLE_TITLES)]
    
    # Article content area
    content_x = x_off + 40
    content_w = min(700, W - x_off - 80)
    
    y = 80
    font_title = get_font(22, bold=True)
    draw.text((content_x, y), title, fill=t["text"], font=font_title)
    bboxes["article_title"] = (content_x, y, content_x+content_w, y+30)
    
    y += 40
    font_meta = get_font(11)
    draw.text((content_x, y), "Published: Nov 28, 2025  |  Author: Alex Chen  |  8 min read",
              fill=t["muted"], font=font_meta)
    bboxes["article_meta"] = (content_x, y, content_x+content_w, y+16)
    
    y += 30
    draw.line([content_x, y, content_x+content_w, y], fill=t["border"])
    y += 16
    
    # Paragraph blocks
    paragraphs = [
        "Machine learning models deployed in production environments face numerous challenges related to data drift, concept drift, and operational reliability. Understanding these challenges is critical for maintaining model performance over time.",
        "In this article, we explore the fundamental principles behind robust model deployment, including monitoring strategies, automated retraining pipelines, and graceful degradation patterns that ensure system reliability.",
        "Key Finding: Models trained on diverse data distributions show 23% better robustness to distribution shift compared to models trained on narrow, curated datasets. This has significant implications for data collection strategies.",
        "Our experiments across 12 different production deployments reveal that early detection of performance degradation, combined with rapid response protocols, reduces the mean time to recovery by approximately 67%.",
    ]
    
    font_body = get_font(13)
    for i, para in enumerate(paragraphs):
        # Word wrap
        words = para.split()
        lines = []
        line = ""
        for w in words:
            test = line + " " + w if line else w
            bbox = font_body.getbbox(test)
            if bbox[2] > content_w - 20:
                lines.append(line)
                line = w
            else:
                line = test
        if line:
            lines.append(line)
        
        for ln in lines:
            draw.text((content_x, y), ln, fill=t["text"], font=font_body)
            y += 20
        
        bboxes[f"paragraph_{i}"] = (content_x, y - len(lines)*20, content_x+content_w, y)
        y += 16
    
    return img, bboxes


def gen_analytics(theme="light", sidebar=True, page_id=0):
    """Generate an analytics page with multiple chart areas."""
    t = THEMES[theme]
    img = Image.new("RGB", (W, H), t["bg"])
    draw = ImageDraw.Draw(img)
    
    bboxes = {}
    x_off = 0
    if sidebar:
        sb_bboxes, x_off = draw_sidebar(draw, theme, NAV_ITEMS, active_idx=1)
        bboxes.update(sb_bboxes)
    
    hdr_bboxes = draw_header(draw, theme, "Analytics Dashboard", x_off)
    bboxes.update(hdr_bboxes)
    
    # Top metrics row
    metrics = [("Users Today", "12,482"), ("Avg. Duration", "3m 45s"),
               ("Pages/Session", "4.8"), ("Bounce Rate", "28.3%")]
    
    font_lbl = get_font(10)
    font_val = get_font(20, bold=True)
    card_w = (W - x_off - 80) // 4
    
    for i, (lbl, val) in enumerate(metrics):
        cx = x_off + 20 + i * (card_w + 12)
        cy = 72
        draw_rounded_rect(draw, (cx, cy, cx+card_w, cy+80), fill=t["card"], radius=8, outline=t["border"])
        draw.text((cx+14, cy+12), lbl, fill=t["muted"], font=font_lbl)
        draw.text((cx+14, cy+34), val, fill=t["text"], font=font_val)
        bboxes[f"analytic_{lbl}"] = (cx, cy, cx+card_w, cy+80)
    
    # Line chart area
    lc_x = x_off + 20
    lc_y = 172
    lc_w = (W - x_off - 50) // 2
    lc_h = 280
    draw_rounded_rect(draw, (lc_x, lc_y, lc_x+lc_w, lc_y+lc_h), fill=t["card"], radius=8, outline=t["border"])
    draw.text((lc_x+16, lc_y+12), "Traffic Over Time", fill=t["text"], font=get_font(13, True))
    
    # Simulated line chart
    pts = np.cumsum(np.random.randn(30) * 5 + 2) + 100
    pts = np.clip(pts, 20, 250)
    for i in range(len(pts)-1):
        x1 = lc_x + 30 + int(i * (lc_w - 60) / 29)
        y1 = lc_y + lc_h - 30 - int(pts[i])
        x2 = lc_x + 30 + int((i+1) * (lc_w - 60) / 29)
        y2 = lc_y + lc_h - 30 - int(pts[i+1])
        draw.line([x1, y1, x2, y2], fill=t["accent"], width=2)
    bboxes["traffic_chart"] = (lc_x, lc_y, lc_x+lc_w, lc_y+lc_h)
    
    # Pie-like donut placeholder
    pc_x = lc_x + lc_w + 12
    pc_y = lc_y
    pc_w = W - pc_x - 20
    draw_rounded_rect(draw, (pc_x, pc_y, pc_x+pc_w, pc_y+lc_h), fill=t["card"], radius=8, outline=t["border"])
    draw.text((pc_x+16, pc_y+12), "Traffic Sources", fill=t["text"], font=get_font(13, True))
    
    # Simulated donut
    center_x = pc_x + pc_w // 2
    center_y = pc_y + lc_h // 2 + 10
    r = 80
    segments = [("Direct", 35, t["accent"]), ("Search", 28, t["success"]),
                ("Social", 22, t["warning"]), ("Referral", 15, t["danger"])]
    start = 0
    font_leg = get_font(10)
    for name, pct, color in segments:
        extent = int(pct * 3.6)
        draw.pieslice([center_x-r, center_y-r, center_x+r, center_y+r],
                     start, start+extent, fill=color)
        start += extent
    # Center hole
    draw.ellipse([center_x-40, center_y-40, center_x+40, center_y+40], fill=t["card"])
    
    # Legend
    ly = pc_y + lc_h - 90
    for name, pct, color in segments:
        draw.rectangle([pc_x+20, ly, pc_x+32, ly+12], fill=color)
        draw.text((pc_x+38, ly), f"{name} ({pct}%)", fill=t["text"], font=font_leg)
        ly += 18
    
    bboxes["source_chart"] = (pc_x, pc_y, pc_x+pc_w, pc_y+lc_h)
    
    return img, bboxes


# ── Page catalog ───────────────────────────────────────────────
PAGE_GENERATORS = [gen_dashboard, gen_settings, gen_table_page, gen_article, gen_analytics]
PAGE_NAMES = ["dashboard", "settings", "table", "article", "analytics"]


# ── Drift transforms ──────────────────────────────────────────
def drift_theme(img, bboxes, orig_theme, page_gen, page_id, sidebar):
    """Change color theme (light->dark, etc.)."""
    alt_themes = [t for t in THEMES if t != orig_theme]
    new_theme = random.choice(alt_themes)
    new_img, new_bboxes = page_gen(theme=new_theme, sidebar=sidebar, page_id=page_id)
    return new_img, new_bboxes, {"type": "theme_change", "from": orig_theme, "to": new_theme, "severity": 1}


def drift_no_sidebar(img, bboxes, orig_theme, page_gen, page_id, sidebar):
    """Remove or add sidebar."""
    new_img, new_bboxes = page_gen(theme=orig_theme, sidebar=not sidebar, page_id=page_id)
    return new_img, new_bboxes, {"type": "sidebar_toggle", "sidebar_removed": sidebar, "severity": 2}


def drift_scale(img, bboxes, *args):
    """Resize/scale the UI (responsive zoom)."""
    scale = random.choice([0.85, 0.9, 1.1, 1.15])
    nw, nh = int(W * scale), int(H * scale)
    resized = img.resize((nw, nh), Image.LANCZOS)
    
    # Center crop/pad to original size
    new_img = Image.new("RGB", (W, H), (200, 200, 200))
    ox = (W - nw) // 2
    oy = (H - nh) // 2
    new_img.paste(resized, (max(0, ox), max(0, oy)))
    
    # Adjust bboxes
    new_bboxes = {}
    for k, (x0, y0, x1, y1) in bboxes.items():
        new_bboxes[k] = (int(x0*scale)+max(0,ox), int(y0*scale)+max(0,oy),
                        int(x1*scale)+max(0,ox), int(y1*scale)+max(0,oy))
    
    return new_img, new_bboxes, {"type": "responsive_scale", "scale": scale, "severity": 3}


def drift_crop(img, bboxes, *args):
    """Random crop simulating viewport shift."""
    cx = random.randint(0, 120)
    cy = random.randint(0, 80)
    cropped = img.crop((cx, cy, W, H))
    new_img = Image.new("RGB", (W, H), img.getpixel((W//2, H//2)))
    new_img.paste(cropped, (0, 0))
    
    new_bboxes = {}
    for k, (x0, y0, x1, y1) in bboxes.items():
        new_bboxes[k] = (max(0, x0-cx), max(0, y0-cy), max(0, x1-cx), max(0, y1-cy))
    
    return new_img, new_bboxes, {"type": "viewport_crop", "offset_x": cx, "offset_y": cy, "severity": 4}


def drift_composite(img, bboxes, orig_theme, page_gen, page_id, sidebar):
    """Composite: theme change + sidebar toggle + slight scale."""
    alt_themes = [t for t in THEMES if t != orig_theme]
    new_theme = random.choice(alt_themes)
    new_img, new_bboxes = page_gen(theme=new_theme, sidebar=not sidebar, page_id=page_id)
    
    # Additional blur
    new_img = new_img.filter(ImageFilter.GaussianBlur(radius=0.5))
    
    # Slight scale
    scale = random.choice([0.92, 0.95, 1.05, 1.08])
    nw, nh = int(W * scale), int(H * scale)
    resized = new_img.resize((nw, nh), Image.LANCZOS)
    final = Image.new("RGB", (W, H), (180, 180, 180))
    ox = (W - nw) // 2
    oy = (H - nh) // 2
    final.paste(resized, (max(0, ox), max(0, oy)))
    
    adj_bboxes = {}
    for k, (x0, y0, x1, y1) in new_bboxes.items():
        adj_bboxes[k] = (int(x0*scale)+max(0,ox), int(y0*scale)+max(0,oy),
                        int(x1*scale)+max(0,ox), int(y1*scale)+max(0,oy))
    
    return final, adj_bboxes, {"type": "composite", "theme": new_theme, "sidebar": not sidebar,
                               "scale": scale, "severity": 5}


DRIFT_FUNCTIONS = [drift_theme, drift_no_sidebar, drift_scale, drift_crop, drift_composite]
DRIFT_NAMES = ["theme_change", "sidebar_toggle", "responsive_scale", "viewport_crop", "composite"]


# ── QA generation ──────────────────────────────────────────────
def generate_qa(page_type, page_id, bboxes):
    """Generate grounded QA pairs for a page."""
    qa_pairs = []
    
    if page_type == "dashboard":
        metrics = DASHBOARD_METRICS[page_id*4:(page_id+1)*4]
        if len(metrics) < 4:
            metrics = DASHBOARD_METRICS[:4]
        for label, value, change in metrics:
            qa_pairs.append({
                "question": f"What is the current value shown for '{label}'?",
                "answer": value,
                "evidence_key": f"metric_{label}",
                "evidence_bbox": list(bboxes.get(f"metric_{label}", (0,0,0,0))),
                "type": "value_extraction"
            })
            qa_pairs.append({
                "question": f"Is the '{label}' metric trending up or down?",
                "answer": "up" if change.startswith("+") else "down",
                "evidence_key": f"metric_{label}",
                "evidence_bbox": list(bboxes.get(f"metric_{label}", (0,0,0,0))),
                "type": "trend_detection"
            })
        # Navigation question
        qa_pairs.append({
            "question": "Which navigation item is currently selected/active?",
            "answer": "Dashboard",
            "evidence_key": "Dashboard",
            "evidence_bbox": list(bboxes.get("Dashboard", (0,0,0,0))),
            "type": "ui_state"
        })
    
    elif page_type == "settings":
        for label, value in SETTINGS_ITEMS:
            qa_pairs.append({
                "question": f"What is the current value for the '{label}' setting?",
                "answer": value,
                "evidence_key": f"setting_{label}",
                "evidence_bbox": list(bboxes.get(f"setting_{label}", (0,0,0,0))),
                "type": "value_extraction"
            })
    
    elif page_type == "table":
        for row in TABLE_DATA[1:]:
            qa_pairs.append({
                "question": f"What is the status of invoice {row[0]}?",
                "answer": row[3],
                "evidence_key": f"tablerow_{row[0]}",
                "evidence_bbox": list(bboxes.get(f"tablerow_{row[0]}", (0,0,0,0))),
                "type": "table_lookup"
            })
            qa_pairs.append({
                "question": f"What is the amount for invoice {row[0]}?",
                "answer": row[2],
                "evidence_key": f"tablerow_{row[0]}",
                "evidence_bbox": list(bboxes.get(f"tablerow_{row[0]}", (0,0,0,0))),
                "type": "table_lookup"
            })
    
    elif page_type == "article":
        title = ARTICLE_TITLES[page_id % len(ARTICLE_TITLES)]
        qa_pairs.append({
            "question": "What is the title of this article?",
            "answer": title,
            "evidence_key": "article_title",
            "evidence_bbox": list(bboxes.get("article_title", (0,0,0,0))),
            "type": "text_extraction"
        })
        qa_pairs.append({
            "question": "Who is the author of this article?",
            "answer": "Alex Chen",
            "evidence_key": "article_meta",
            "evidence_bbox": list(bboxes.get("article_meta", (0,0,0,0))),
            "type": "text_extraction"
        })
        qa_pairs.append({
            "question": "What percentage improvement in robustness is mentioned for models trained on diverse data?",
            "answer": "23%",
            "evidence_key": "paragraph_2",
            "evidence_bbox": list(bboxes.get("paragraph_2", (0,0,0,0))),
            "type": "value_extraction"
        })
    
    elif page_type == "analytics":
        metrics = [("Users Today", "12,482"), ("Avg. Duration", "3m 45s"),
                   ("Pages/Session", "4.8"), ("Bounce Rate", "28.3%")]
        for lbl, val in metrics:
            qa_pairs.append({
                "question": f"What is the value displayed for '{lbl}'?",
                "answer": val,
                "evidence_key": f"analytic_{lbl}",
                "evidence_bbox": list(bboxes.get(f"analytic_{lbl}", (0,0,0,0))),
                "type": "value_extraction"
            })
        qa_pairs.append({
            "question": "What is the largest traffic source shown in the chart?",
            "answer": "Direct (35%)",
            "evidence_key": "source_chart",
            "evidence_bbox": list(bboxes.get("source_chart", (0,0,0,0))),
            "type": "chart_reading"
        })
    
    return qa_pairs


# ── Main generation ────────────────────────────────────────────
def main():
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    
    benchmark = {"pages": [], "total_images": 0, "total_qa_pairs": 0}
    
    n_pages = 50
    page_id = 0
    
    for i in range(n_pages):
        gen_idx = i % len(PAGE_GENERATORS)
        page_gen = PAGE_GENERATORS[gen_idx]
        page_type = PAGE_NAMES[gen_idx]
        sub_id = i // len(PAGE_GENERATORS)
        
        # Generate base image
        theme = "light"
        sidebar = True
        base_img, base_bboxes = page_gen(theme=theme, sidebar=sidebar, page_id=sub_id)
        
        base_fname = f"page_{i:03d}_base.png"
        base_img.save(IMG_DIR / base_fname, "PNG")
        
        # Generate QA
        qa_pairs = generate_qa(page_type, sub_id, base_bboxes)
        
        page_entry = {
            "page_id": i,
            "page_type": page_type,
            "base_image": base_fname,
            "base_theme": theme,
            "base_sidebar": sidebar,
            "base_bboxes": {k: list(v) for k, v in base_bboxes.items()},
            "qa_pairs": qa_pairs,
            "variants": []
        }
        
        # Generate 5 drift variants
        for d, (drift_fn, drift_name) in enumerate(zip(DRIFT_FUNCTIONS, DRIFT_NAMES)):
            try:
                drift_img, drift_bboxes, drift_meta = drift_fn(
                    base_img.copy(), base_bboxes.copy(), theme, page_gen, sub_id, sidebar
                )
                variant_fname = f"page_{i:03d}_drift_{drift_name}.png"
                drift_img.save(IMG_DIR / variant_fname, "PNG")
                
                page_entry["variants"].append({
                    "variant_id": d,
                    "drift_type": drift_name,
                    "image": variant_fname,
                    "drift_meta": drift_meta,
                    "severity": drift_meta["severity"],
                    "bboxes": {k: list(v) for k, v in drift_bboxes.items()},
                })
            except Exception as e:
                print(f"Warning: drift {drift_name} failed for page {i}: {e}")
        
        benchmark["pages"].append(page_entry)
        benchmark["total_images"] += 1 + len(page_entry["variants"])
        benchmark["total_qa_pairs"] += len(qa_pairs)
        
        if (i + 1) % 10 == 0:
            print(f"Generated {i+1}/{n_pages} pages...")
    
    # Save benchmark metadata
    with open(META_FILE, "w") as f:
        json.dump(benchmark, f, indent=2)
    
    print(f"\nBenchmark generated:")
    print(f"  Pages: {len(benchmark['pages'])}")
    print(f"  Total images: {benchmark['total_images']}")
    print(f"  Total QA pairs: {benchmark['total_qa_pairs']}")
    print(f"  Output: {IMG_DIR}")


if __name__ == "__main__":
    main()
