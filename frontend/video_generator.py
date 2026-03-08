"""Protein analysis video generator.

Renders an MP4 video showing a 3D protein structure rotating with
risk annotations, danger zone zoom-in, and stats overlay.
Uses matplotlib 3D plotting + ffmpeg — no GPU or display server needed.
"""

import io
import tempfile
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")  # headless backend

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from matplotlib.patches import FancyBboxPatch


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


def parse_pdb_backbone(pdb_string: str) -> dict:
    """Extract C-alpha coordinates and B-factors (pLDDT) from PDB string."""
    ca_coords = []
    residue_numbers = []
    bfactors = []
    residue_names = []

    for line in pdb_string.splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        atom_name = line[12:16].strip()
        if atom_name != "CA":
            continue

        x = float(line[30:38].strip())
        y = float(line[38:46].strip())
        z = float(line[46:54].strip())
        bfactor = float(line[60:66].strip())
        resnum = int(line[22:26].strip())
        resname = line[17:20].strip()

        ca_coords.append([x, y, z])
        residue_numbers.append(resnum)
        bfactors.append(bfactor)
        residue_names.append(resname)

    coords = np.array(ca_coords)
    # Center the structure at origin
    center = coords.mean(axis=0)
    coords -= center

    return {
        "coords": coords,
        "residue_numbers": residue_numbers,
        "bfactors": np.array(bfactors),
        "residue_names": residue_names,
        "center": center,
    }


def _get_risk_color(risk_level: str) -> str:
    """Get color for risk level."""
    return {"HIGH": "#ef4444", "MEDIUM": "#f59e0b", "LOW": "#22c55e"}.get(risk_level, "#94a3b8")


def _rotation_matrix(angle_deg: float, axis: str = "y") -> np.ndarray:
    """Create a 3D rotation matrix around the given axis."""
    theta = np.radians(angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    elif axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    else:
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _color_backbone(
    backbone: dict,
    pocket_residues: list[int],
    danger_residues: list[int],
) -> np.ndarray:
    """Assign colors to each residue based on classification."""
    n = len(backbone["residue_numbers"])
    colors = np.zeros((n, 4))

    pocket_set = set(pocket_residues)
    danger_set = set(danger_residues)

    for i, resnum in enumerate(backbone["residue_numbers"]):
        if resnum in danger_set:
            colors[i] = [0.94, 0.27, 0.27, 1.0]  # red
        elif resnum in pocket_set:
            colors[i] = [0.96, 0.62, 0.04, 1.0]  # orange
        else:
            plddt = backbone["bfactors"][i]
            # Blue-white gradient based on pLDDT (50-100 range)
            t = np.clip((plddt - 50) / 50, 0, 1)
            colors[i] = [0.53 - 0.33 * t, 0.65 - 0.15 * t, 0.85 + 0.15 * t, 0.85]

    return colors


def _draw_stats_panel(fig, video_data: ProteinVideoData, phase: str, alpha: float = 1.0):
    """Draw a translucent stats overlay panel on the figure."""
    if alpha <= 0:
        return

    risk_color = _get_risk_color(video_data.risk_level)
    factors = video_data.risk_factors
    emb_sim = factors.get("max_embedding_similarity", 0)
    struct_sim = factors.get("max_structure_similarity")
    func_overlap = factors.get("function_overlap", 0)

    top_match = video_data.top_matches[0] if video_data.top_matches else {}
    match_name = top_match.get("name", "No match")
    match_org = top_match.get("organism", "")

    # Create an overlay axes for stats
    ax_stats = fig.add_axes([0.02, 0.02, 0.35, 0.55], facecolor="none")
    ax_stats.set_xlim(0, 1)
    ax_stats.set_ylim(0, 1)
    ax_stats.axis("off")

    # Semi-transparent background
    bg = FancyBboxPatch(
        (0.02, 0.02), 0.96, 0.96,
        boxstyle="round,pad=0.03",
        facecolor=(0.05, 0.05, 0.12, 0.85 * alpha),
        edgecolor=(0.3, 0.3, 0.4, 0.6 * alpha),
        linewidth=1.5,
    )
    ax_stats.add_patch(bg)

    text_alpha = alpha
    y = 0.90

    # Title
    ax_stats.text(0.5, y, "SCREENING RESULTS", ha="center", va="top",
                  fontsize=9, fontweight="bold", color=(0.7, 0.8, 1.0, text_alpha),
                  fontfamily="monospace")
    y -= 0.11

    # Risk score
    ax_stats.text(0.08, y, "Risk Score", va="top", fontsize=8,
                  color=(0.7, 0.7, 0.8, text_alpha), fontfamily="monospace")
    ax_stats.text(0.92, y, f"{video_data.risk_score:.3f}", va="top", ha="right",
                  fontsize=14, fontweight="bold", color=risk_color,
                  fontfamily="monospace", alpha=text_alpha)
    y -= 0.09
    ax_stats.text(0.08, y, f"{video_data.risk_level}", va="top", fontsize=8,
                  fontweight="bold", color=risk_color,
                  fontfamily="monospace", alpha=text_alpha)
    y -= 0.10

    # Divider
    ax_stats.plot([0.08, 0.92], [y + 0.03, y + 0.03],
                  color=(0.3, 0.3, 0.5, 0.5 * alpha), linewidth=0.8)

    # Top match
    ax_stats.text(0.08, y, "Top Match", va="top", fontsize=7,
                  color=(0.6, 0.6, 0.7, text_alpha), fontfamily="monospace")
    y -= 0.08
    display_name = match_name[:22] + "..." if len(match_name) > 25 else match_name
    ax_stats.text(0.08, y, display_name, va="top", fontsize=7.5,
                  fontweight="bold", color=(0.9, 0.9, 1.0, text_alpha),
                  fontfamily="monospace")
    y -= 0.07
    if match_org:
        display_org = match_org[:25] + "..." if len(match_org) > 28 else match_org
        ax_stats.text(0.08, y, display_org, va="top", fontsize=6.5,
                      fontstyle="italic", color=(0.5, 0.5, 0.6, text_alpha),
                      fontfamily="monospace")
        y -= 0.09

    # Similarity bars
    bars = [("Embedding", emb_sim)]
    if struct_sim is not None:
        bars.append(("Structure", struct_sim))
    bars.append(("Function", func_overlap))

    for label, val in bars:
        ax_stats.text(0.08, y, label, va="top", fontsize=6.5,
                      color=(0.6, 0.6, 0.7, text_alpha), fontfamily="monospace")
        ax_stats.text(0.92, y, f"{val:.3f}", va="top", ha="right", fontsize=6.5,
                      color=(0.9, 0.9, 1.0, text_alpha), fontfamily="monospace")
        y -= 0.06
        # Progress bar
        bar_y = y + 0.02
        bar_h = 0.025
        # Background
        bg_bar = FancyBboxPatch(
            (0.08, bar_y), 0.84, bar_h,
            boxstyle="round,pad=0.005",
            facecolor=(0.15, 0.15, 0.25, 0.6 * alpha),
            edgecolor="none",
        )
        ax_stats.add_patch(bg_bar)
        # Fill
        fill_w = max(0.01, val * 0.84)
        bar_color = _get_risk_color("HIGH") if val > 0.7 else ("#f59e0b" if val > 0.4 else "#22c55e")
        fill_bar = FancyBboxPatch(
            (0.08, bar_y), fill_w, bar_h,
            boxstyle="round,pad=0.005",
            facecolor=(*matplotlib.colors.to_rgb(bar_color), 0.8 * alpha),
            edgecolor="none",
        )
        ax_stats.add_patch(fill_bar)
        y -= 0.07

    # Sequence length
    ax_stats.text(0.08, y, f"Length: {video_data.sequence_length} aa", va="top",
                  fontsize=6.5, color=(0.5, 0.5, 0.6, text_alpha), fontfamily="monospace")


def _draw_title_card(fig, video_data: ProteinVideoData, alpha: float = 1.0):
    """Draw a title/intro overlay."""
    if alpha <= 0:
        return

    risk_color = _get_risk_color(video_data.risk_level)

    ax_title = fig.add_axes([0.0, 0.0, 1.0, 1.0], facecolor="none")
    ax_title.set_xlim(0, 1)
    ax_title.set_ylim(0, 1)
    ax_title.axis("off")

    # Background overlay
    bg = FancyBboxPatch(
        (0.15, 0.25), 0.70, 0.50,
        boxstyle="round,pad=0.04",
        facecolor=(0.05, 0.05, 0.12, 0.92 * alpha),
        edgecolor=(0.3, 0.3, 0.5, 0.6 * alpha),
        linewidth=2,
    )
    ax_title.add_patch(bg)

    ax_title.text(0.5, 0.68, "BIOSCREEN", ha="center", va="center",
                  fontsize=22, fontweight="bold", color=(0.5, 0.7, 1.0, alpha),
                  fontfamily="monospace")
    ax_title.text(0.5, 0.60, "Protein Risk Analysis", ha="center", va="center",
                  fontsize=12, color=(0.7, 0.7, 0.8, alpha), fontfamily="monospace")

    seq_id = video_data.top_matches[0].get("name", "Query") if video_data.top_matches else "Query"
    ax_title.text(0.5, 0.48, f"Sequence: {video_data.sequence_length} amino acids",
                  ha="center", va="center", fontsize=9,
                  color=(0.6, 0.6, 0.7, alpha), fontfamily="monospace")

    ax_title.text(0.5, 0.38, f"Risk: {video_data.risk_level}", ha="center", va="center",
                  fontsize=16, fontweight="bold", color=risk_color,
                  fontfamily="monospace", alpha=alpha)

    mode = "Full (Structure + Embedding + Function)" if video_data.structure_predicted else "Fast (Embedding + Function)"
    ax_title.text(0.5, 0.30, f"Mode: {mode}", ha="center", va="center",
                  fontsize=7.5, color=(0.5, 0.5, 0.6, alpha), fontfamily="monospace")


def _draw_danger_label(fig, video_data: ProteinVideoData, alpha: float = 1.0):
    """Draw a label indicating the danger zone focus."""
    if alpha <= 0 or not video_data.danger_residues:
        return

    ax_label = fig.add_axes([0.55, 0.78, 0.43, 0.18], facecolor="none")
    ax_label.set_xlim(0, 1)
    ax_label.set_ylim(0, 1)
    ax_label.axis("off")

    bg = FancyBboxPatch(
        (0.02, 0.02), 0.96, 0.96,
        boxstyle="round,pad=0.03",
        facecolor=(0.3, 0.05, 0.05, 0.85 * alpha),
        edgecolor=(0.94, 0.27, 0.27, 0.7 * alpha),
        linewidth=1.5,
    )
    ax_label.add_patch(bg)

    ax_label.text(0.5, 0.65, "DANGER ZONE", ha="center", va="center",
                  fontsize=10, fontweight="bold", color=(1.0, 0.3, 0.3, alpha),
                  fontfamily="monospace")
    ax_label.text(0.5, 0.30, f"{len(video_data.danger_residues)} residues match toxin active site",
                  ha="center", va="center", fontsize=7,
                  color=(0.9, 0.6, 0.6, alpha), fontfamily="monospace")


def _draw_verdict(fig, video_data: ProteinVideoData, alpha: float = 1.0):
    """Draw the final verdict card."""
    if alpha <= 0:
        return

    risk_color = _get_risk_color(video_data.risk_level)
    explanation = video_data.risk_factors.get("score_explanation", "")
    parts = [p.strip() for p in explanation.split(". ") if p.strip()]
    verdict_text = parts[0] if parts else f"{video_data.risk_level} RISK"

    ax_v = fig.add_axes([0.0, 0.0, 1.0, 1.0], facecolor="none")
    ax_v.set_xlim(0, 1)
    ax_v.set_ylim(0, 1)
    ax_v.axis("off")

    bg = FancyBboxPatch(
        (0.20, 0.35), 0.60, 0.30,
        boxstyle="round,pad=0.04",
        facecolor=(0.05, 0.05, 0.12, 0.92 * alpha),
        edgecolor=(*matplotlib.colors.to_rgb(risk_color), 0.7 * alpha),
        linewidth=2,
    )
    ax_v.add_patch(bg)

    ax_v.text(0.5, 0.58, "VERDICT", ha="center", va="center",
              fontsize=10, color=(0.6, 0.6, 0.7, alpha), fontfamily="monospace")

    # Truncate verdict if too long
    if len(verdict_text) > 50:
        verdict_text = verdict_text[:47] + "..."

    ax_v.text(0.5, 0.48, verdict_text, ha="center", va="center",
              fontsize=9, fontweight="bold", color=risk_color,
              fontfamily="monospace", alpha=alpha)

    ax_v.text(0.5, 0.40, f"Score: {video_data.risk_score:.3f}",
              ha="center", va="center", fontsize=12, fontweight="bold",
              color=risk_color, fontfamily="monospace", alpha=alpha)


def generate_video(video_data: ProteinVideoData, fps: int = 24, duration: float = 12.0) -> bytes:
    """Generate an MP4 video of the protein analysis.

    Video timeline:
      0-2s   : Title card fades in/out over structure
      2-7s   : Full 360-degree rotation with stats panel
      7-9.5s : Zoom into danger zone (or continue rotation if no danger)
      9.5-12s: Final verdict overlay

    Args:
        video_data: All screening data needed for the video.
        fps: Frames per second (default 24).
        duration: Total video duration in seconds.

    Returns:
        MP4 video as bytes.
    """
    backbone = parse_pdb_backbone(video_data.pdb_string)
    coords = backbone["coords"]
    colors = _color_backbone(backbone, video_data.pocket_residues, video_data.danger_residues)

    total_frames = int(fps * duration)

    # Precompute danger zone center for zoom
    danger_set = set(video_data.danger_residues)
    danger_mask = np.array([r in danger_set for r in backbone["residue_numbers"]])
    if danger_mask.any():
        danger_center = coords[danger_mask].mean(axis=0)
    else:
        danger_center = np.zeros(3)

    # Figure setup — dark background
    fig = plt.figure(figsize=(10, 6), dpi=120, facecolor="#0a0a14")
    ax = fig.add_axes([0.05, 0.05, 0.9, 0.9], projection="3d", facecolor="#0a0a14")

    # Compute a reasonable view distance
    max_extent = np.abs(coords).max() * 1.3

    def init():
        return []

    def animate(frame_idx):
        fig.clf()
        ax_3d = fig.add_axes([0.05, 0.05, 0.9, 0.9], projection="3d", facecolor="#0a0a14")
        fig.patch.set_facecolor("#0a0a14")

        t = frame_idx / total_frames  # 0 to 1 progress
        t_sec = t * duration  # time in seconds

        # Camera angle — continuous rotation
        azim = -60 + (frame_idx / total_frames) * 360
        elev = 15 + 10 * np.sin(t * np.pi * 2)

        # Zoom phases
        if t_sec < 2.0:
            # Title phase — normal view
            zoom = max_extent
        elif t_sec < 7.0:
            # Rotation phase — normal view
            zoom = max_extent
        elif t_sec < 9.5 and danger_mask.any():
            # Zoom into danger zone
            zoom_t = (t_sec - 7.0) / 2.5
            zoom_t = 3 * zoom_t**2 - 2 * zoom_t**3  # smooth ease
            zoom = max_extent * (1.0 - 0.55 * zoom_t)
            # Shift view center toward danger zone
            view_center = danger_center * zoom_t
            shifted = coords - view_center
        else:
            zoom = max_extent

        # Apply rotation to coordinates
        rot = _rotation_matrix(azim, "y") @ _rotation_matrix(elev, "x")

        if t_sec >= 7.0 and t_sec < 9.5 and danger_mask.any():
            rotated = (rot @ shifted.T).T
        else:
            rotated = (rot @ coords.T).T

        # Draw backbone as connected line segments with per-segment coloring
        for i in range(len(rotated) - 1):
            ax_3d.plot(
                [rotated[i, 0], rotated[i + 1, 0]],
                [rotated[i, 1], rotated[i + 1, 1]],
                [rotated[i, 2], rotated[i + 1, 2]],
                color=colors[i],
                linewidth=2.0,
                alpha=float(colors[i, 3]),
                solid_capstyle="round",
            )

        # Draw danger residues as larger spheres
        if danger_mask.any():
            danger_pts = rotated[danger_mask]
            if t_sec >= 7.0 and t_sec < 9.5:
                # Pulsing glow during zoom
                pulse = 0.5 + 0.5 * np.sin(t_sec * 6)
                s = 80 + 40 * pulse
            else:
                s = 50
            ax_3d.scatter(
                danger_pts[:, 0], danger_pts[:, 1], danger_pts[:, 2],
                c="#ef4444", s=s, alpha=0.8, edgecolors="#ff6b6b",
                linewidth=0.5, depthshade=True,
            )

        # Draw pocket residues as medium spheres
        pocket_set = set(video_data.pocket_residues)
        pocket_mask = np.array([r in pocket_set for r in backbone["residue_numbers"]])
        if pocket_mask.any():
            pocket_pts = rotated[pocket_mask]
            ax_3d.scatter(
                pocket_pts[:, 0], pocket_pts[:, 1], pocket_pts[:, 2],
                c="#f59e0b", s=35, alpha=0.6, edgecolors="#fbbf24",
                linewidth=0.3, depthshade=True,
            )

        # Set view limits
        ax_3d.set_xlim(-zoom, zoom)
        ax_3d.set_ylim(-zoom, zoom)
        ax_3d.set_zlim(-zoom, zoom)
        ax_3d.axis("off")
        ax_3d.set_box_aspect([1, 1, 1])

        # --- Overlays based on timeline phase ---

        # Title card (0-2s, fade in 0-0.5s, hold, fade out 1.5-2s)
        if t_sec < 2.0:
            if t_sec < 0.5:
                title_alpha = t_sec / 0.5
            elif t_sec > 1.5:
                title_alpha = (2.0 - t_sec) / 0.5
            else:
                title_alpha = 1.0
            _draw_title_card(fig, video_data, alpha=title_alpha)

        # Stats panel (2-9.5s, fade in 2-2.5s, fade out 9-9.5s)
        if 2.0 <= t_sec <= 9.5:
            if t_sec < 2.5:
                stats_alpha = (t_sec - 2.0) / 0.5
            elif t_sec > 9.0:
                stats_alpha = (9.5 - t_sec) / 0.5
            else:
                stats_alpha = 1.0
            _draw_stats_panel(fig, video_data, phase="rotation", alpha=stats_alpha)

        # Danger zone label (7-9.5s)
        if 7.0 <= t_sec < 9.5 and video_data.danger_residues:
            if t_sec < 7.5:
                danger_alpha = (t_sec - 7.0) / 0.5
            elif t_sec > 9.0:
                danger_alpha = (9.5 - t_sec) / 0.5
            else:
                danger_alpha = 1.0
            _draw_danger_label(fig, video_data, alpha=danger_alpha)

        # Final verdict (9.5-12s, fade in 9.5-10s)
        if t_sec >= 9.5:
            if t_sec < 10.0:
                verdict_alpha = (t_sec - 9.5) / 0.5
            else:
                verdict_alpha = 1.0
            _draw_verdict(fig, video_data, alpha=verdict_alpha)

        # Watermark
        fig.text(0.98, 0.02, "BioScreen", ha="right", va="bottom",
                 fontsize=7, color=(0.3, 0.3, 0.4, 0.6), fontfamily="monospace")

        return []

    anim = animation.FuncAnimation(
        fig, animate, init_func=init,
        frames=total_frames, interval=1000 / fps, blit=False,
    )

    # Write to temporary file, then read bytes
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as tmp:
        writer = animation.FFMpegWriter(
            fps=fps,
            codec="libx264",
            extra_args=["-pix_fmt", "yuv420p", "-preset", "fast"],
        )
        anim.save(tmp.name, writer=writer)
        plt.close(fig)
        tmp.seek(0)
        return tmp.read()
