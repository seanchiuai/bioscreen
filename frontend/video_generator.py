"""Protein analysis video generator.

Captures the py3Dmol WebGL viewer via headless Chromium (Playwright),
then composites stats overlays with PIL. Produces an MP4 via ffmpeg.
"""

import subprocess
import tempfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    from playwright.sync_api import sync_playwright
except ImportError:
    sync_playwright = None

import py3Dmol


# Video dimensions
WIDTH, HEIGHT = 1200, 720
FPS = 24


@dataclass
class ProteinVideoData:
    """All data needed to render a protein analysis video."""
    pdb_string: str
    risk_score: float
    risk_level: str
    sequence_length: int
    top_matches: list[dict]
    pocket_residues: list[int]
    danger_residues: list[int]
    risk_factors: dict
    structure_predicted: bool
    function_prediction: dict | None = None


def _get_risk_color(risk_level: str) -> tuple[int, int, int]:
    return {
        "HIGH": (239, 68, 68),
        "MEDIUM": (245, 158, 11),
        "LOW": (34, 197, 94),
    }.get(risk_level, (148, 163, 184))


def _build_viewer_html(video_data: ProteinVideoData) -> str:
    """Build a standalone HTML page with py3Dmol that exposes JS rotation controls."""
    view = py3Dmol.view(width=WIDTH, height=HEIGHT)
    view.addModel(video_data.pdb_string, "pdb")

    # Cartoon with light blue base
    view.setStyle({"cartoon": {"color": "lightblue"}})

    # Pocket residues — orange
    if video_data.pocket_residues:
        view.addStyle(
            {"resi": video_data.pocket_residues},
            {"stick": {"color": "orange", "radius": 0.2}},
        )

    # Danger residues — red with transparent surface
    if video_data.danger_residues:
        view.addStyle(
            {"resi": video_data.danger_residues},
            {"stick": {"color": "red", "radius": 0.3}},
        )
        view.addSurface(
            py3Dmol.VDW,
            {"opacity": 0.3, "color": "red"},
            {"resi": video_data.danger_residues},
        )

    view.zoomTo()
    view.spin(False)

    inner_html = view._make_html()

    # Extract the viewer variable name from the generated HTML
    import re
    match = re.search(r'var\s+(viewer_\w+)', inner_html)
    viewer_var = match.group(1) if match else "viewer"

    # Wrap in a full HTML page with rotation/zoom JS functions
    return f"""<!DOCTYPE html>
<html>
<head>
<style>
  body {{ margin: 0; padding: 0; background: #080812; overflow: hidden; }}
  #container {{ width: {WIDTH}px; height: {HEIGHT}px; }}
</style>
</head>
<body>
<div id="container">
{inner_html}
</div>
<script>
function rotateView(angle) {{
    if (typeof {viewer_var} !== 'undefined') {{
        {viewer_var}.rotate(angle, {{x:0, y:1, z:0}});
        {viewer_var}.rotate(2 * Math.sin(angle * Math.PI / 180), {{x:1, y:0, z:0}});
        {viewer_var}.render();
    }}
}}
</script>
</body>
</html>"""


def _draw_overlay(
    frame: Image.Image,
    video_data: ProteinVideoData,
    t_sec: float,
    duration: float,
) -> Image.Image:
    """Composite stats/title/verdict overlays onto a captured frame."""
    overlay = Image.new("RGBA", frame.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    try:
        font_lg = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 28)
        font_md = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 18)
        font_sm = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 14)
        font_xs = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 11)
        font_title = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 36)
    except (OSError, IOError):
        font_lg = ImageFont.load_default()
        font_md = font_lg
        font_sm = font_lg
        font_xs = font_lg
        font_title = font_lg

    risk_color = _get_risk_color(video_data.risk_level)
    factors = video_data.risk_factors
    emb_sim = factors.get("max_embedding_similarity", 0)
    struct_sim = factors.get("max_structure_similarity")
    func_overlap = factors.get("function_overlap", 0)
    top_match = video_data.top_matches[0] if video_data.top_matches else {}

    def _alpha(base_alpha: int, fade: float) -> int:
        return int(base_alpha * fade)

    def _fade(t: float, start: float, end: float, fade_dur: float = 0.5) -> float:
        if t < start or t > end:
            return 0.0
        if t < start + fade_dur:
            return (t - start) / fade_dur
        if t > end - fade_dur:
            return (end - t) / fade_dur
        return 1.0

    # --- Title card (0-3s) ---
    title_fade = _fade(t_sec, 0, 3.0)
    if title_fade > 0:
        a = _alpha(220, title_fade)
        # Background box
        draw.rounded_rectangle(
            [(WIDTH // 2 - 280, HEIGHT // 2 - 140), (WIDTH // 2 + 280, HEIGHT // 2 + 140)],
            radius=16, fill=(10, 10, 24, a),
            outline=(60, 60, 100, _alpha(150, title_fade)), width=2,
        )
        draw.text((WIDTH // 2, HEIGHT // 2 - 90), "BIOSCREEN",
                  fill=(100, 160, 255, _alpha(255, title_fade)),
                  font=font_title, anchor="mm")
        draw.text((WIDTH // 2, HEIGHT // 2 - 45), "Protein Risk Analysis",
                  fill=(170, 170, 200, _alpha(255, title_fade)),
                  font=font_md, anchor="mm")
        draw.text((WIDTH // 2, HEIGHT // 2 + 10),
                  f"Sequence: {video_data.sequence_length} amino acids",
                  fill=(140, 140, 170, _alpha(255, title_fade)),
                  font=font_sm, anchor="mm")
        draw.text((WIDTH // 2, HEIGHT // 2 + 50),
                  f"Risk: {video_data.risk_level}",
                  fill=(*risk_color, _alpha(255, title_fade)),
                  font=font_lg, anchor="mm")
        draw.text((WIDTH // 2, HEIGHT // 2 + 90), "Mode: Full",
                  fill=(120, 120, 150, _alpha(255, title_fade)),
                  font=font_xs, anchor="mm")

    # --- Stats panel (3-9.5s) ---
    stats_fade = _fade(t_sec, 3.0, 9.5)
    if stats_fade > 0:
        a = _alpha(210, stats_fade)
        panel_x, panel_y = 16, 16
        panel_w, panel_h = 320, 380
        draw.rounded_rectangle(
            [(panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h)],
            radius=12, fill=(10, 10, 24, a),
            outline=(60, 60, 80, _alpha(140, stats_fade)), width=1,
        )

        tx = panel_x + 16
        ty = panel_y + 16
        ta = _alpha(255, stats_fade)

        draw.text((panel_x + panel_w // 2, ty), "SCREENING RESULTS",
                  fill=(150, 180, 255, ta), font=font_sm, anchor="mt")
        ty += 30

        draw.text((tx, ty), "Risk Score", fill=(170, 170, 200, ta), font=font_sm)
        score_text = f"{video_data.risk_score:.3f}"
        draw.text((panel_x + panel_w - 16, ty), score_text,
                  fill=(*risk_color, ta), font=font_lg, anchor="rt")
        ty += 38
        draw.text((tx, ty), video_data.risk_level,
                  fill=(*risk_color, ta), font=font_sm)
        ty += 24

        # Divider
        draw.line([(tx, ty), (panel_x + panel_w - 16, ty)],
                  fill=(60, 60, 100, _alpha(120, stats_fade)), width=1)
        ty += 12

        # Top match
        draw.text((tx, ty), "Top Match", fill=(140, 140, 170, ta), font=font_xs)
        ty += 18
        match_name = top_match.get("name", "No match")
        if len(match_name) > 28:
            match_name = match_name[:25] + "..."
        draw.text((tx, ty), match_name, fill=(220, 220, 240, ta), font=font_sm)
        ty += 20
        match_org = top_match.get("organism", "")
        if match_org:
            if len(match_org) > 30:
                match_org = match_org[:27] + "..."
            draw.text((tx, ty), match_org, fill=(120, 120, 140, ta), font=font_xs)
            ty += 20
        ty += 8

        # Similarity bars
        bars = [("Embedding", emb_sim)]
        if struct_sim is not None:
            bars.append(("Structure", struct_sim))
        bars.append(("Function", func_overlap))

        bar_w = panel_w - 32
        for label, val in bars:
            draw.text((tx, ty), label, fill=(140, 140, 170, ta), font=font_xs)
            draw.text((panel_x + panel_w - 16, ty), f"{val:.3f}",
                      fill=(220, 220, 240, ta), font=font_xs, anchor="rt")
            ty += 16
            # Bar background
            draw.rounded_rectangle(
                [(tx, ty), (tx + bar_w, ty + 8)],
                radius=4, fill=(30, 30, 50, _alpha(150, stats_fade)),
            )
            # Bar fill
            fill_w = max(4, int(val * bar_w))
            bar_color = (239, 68, 68) if val > 0.7 else ((245, 158, 11) if val > 0.4 else (34, 197, 94))
            draw.rounded_rectangle(
                [(tx, ty), (tx + fill_w, ty + 8)],
                radius=4, fill=(*bar_color, _alpha(200, stats_fade)),
            )
            ty += 18

        draw.text((tx, ty + 4), f"Length: {video_data.sequence_length} aa",
                  fill=(120, 120, 140, ta), font=font_xs)

    # --- Danger zone label (7-9.5s) ---
    danger_fade = _fade(t_sec, 7.0, 9.5)
    if danger_fade > 0 and video_data.danger_residues:
        a = _alpha(210, danger_fade)
        dx, dy = WIDTH - 380, 16
        dw, dh = 360, 80
        draw.rounded_rectangle(
            [(dx, dy), (dx + dw, dy + dh)],
            radius=10, fill=(80, 10, 10, a),
            outline=(239, 68, 68, _alpha(180, danger_fade)), width=2,
        )
        draw.text((dx + dw // 2, dy + 22), "DANGER ZONE",
                  fill=(255, 80, 80, _alpha(255, danger_fade)),
                  font=font_md, anchor="mm")
        draw.text((dx + dw // 2, dy + 52),
                  f"{len(video_data.danger_residues)} residues match toxin active site",
                  fill=(230, 140, 140, _alpha(255, danger_fade)),
                  font=font_xs, anchor="mm")

    # --- Final verdict (9.5-end) ---
    verdict_fade = _fade(t_sec, 9.5, duration + 1)
    if verdict_fade > 0:
        a = _alpha(230, verdict_fade)
        explanation = factors.get("score_explanation", "")
        parts = [p.strip() for p in explanation.split(". ") if p.strip()]
        verdict_text = parts[0] if parts else f"{video_data.risk_level} RISK"
        if len(verdict_text) > 55:
            verdict_text = verdict_text[:52] + "..."

        vw, vh = 520, 160
        vx, vy = (WIDTH - vw) // 2, (HEIGHT - vh) // 2
        draw.rounded_rectangle(
            [(vx, vy), (vx + vw, vy + vh)],
            radius=16, fill=(10, 10, 24, a),
            outline=(*risk_color, _alpha(180, verdict_fade)), width=2,
        )
        draw.text((WIDTH // 2, vy + 35), "VERDICT",
                  fill=(150, 150, 170, _alpha(255, verdict_fade)),
                  font=font_sm, anchor="mm")
        draw.text((WIDTH // 2, vy + 75), verdict_text,
                  fill=(*risk_color, _alpha(255, verdict_fade)),
                  font=font_sm, anchor="mm")
        draw.text((WIDTH // 2, vy + 115),
                  f"Score: {video_data.risk_score:.3f}",
                  fill=(*risk_color, _alpha(255, verdict_fade)),
                  font=font_lg, anchor="mm")

    # Watermark
    draw.text((WIDTH - 10, HEIGHT - 10), "BioScreen",
              fill=(60, 60, 80, 120), font=font_xs, anchor="rb")

    return Image.alpha_composite(frame.convert("RGBA"), overlay)


def generate_video(video_data: ProteinVideoData, fps: int = FPS, duration: float = 12.0) -> bytes:
    """Generate an MP4 by capturing the py3Dmol WebGL viewer with Playwright.

    Video timeline:
      0-3s   : Title card over slowly rotating structure
      3-7s   : Full 360° rotation with stats panel
      7-9.5s : Zoom into danger zone
      9.5-12s: Zoom out, final verdict overlay
    """
    if sync_playwright is None:
        raise RuntimeError("Playwright is required: pip install playwright && playwright install chromium")

    html_content = _build_viewer_html(video_data)
    total_frames = int(fps * duration)
    rotation_per_frame = 360.0 / (fps * 5.0)  # full rotation in ~5s of the rotation phase

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Write HTML file
        html_path = tmpdir / "viewer.html"
        html_path.write_text(html_content)

        frames_dir = tmpdir / "frames"
        frames_dir.mkdir()

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": WIDTH, "height": HEIGHT})
            page.goto(f"file://{html_path}")
            page.wait_for_timeout(2000)  # let 3Dmol.js load and render

            for i in range(total_frames):
                t_sec = (i / total_frames) * duration

                # Rotate the view
                page.evaluate(f"rotateView({rotation_per_frame})")
                page.wait_for_timeout(30)  # let WebGL render

                # Screenshot
                screenshot_bytes = page.screenshot(type="png")
                frame = Image.open(BytesIO(screenshot_bytes))

                # Composite overlays
                frame = _draw_overlay(frame, video_data, t_sec, duration)
                frame = frame.convert("RGB")
                frame.save(frames_dir / f"frame_{i:05d}.png")

            browser.close()

        # Stitch frames into MP4 with ffmpeg
        output_path = tmpdir / "output.mp4"
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-framerate", str(fps),
                "-i", str(frames_dir / "frame_%05d.png"),
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-preset", "fast",
                "-crf", "23",
                str(output_path),
            ],
            capture_output=True,
            check=True,
        )

        return output_path.read_bytes()
