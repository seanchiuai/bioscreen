"""Protein analysis video generator.

Renders an MP4 video showing a 3D protein structure rotating with
risk annotations, danger zone zoom-in, and stats overlay.
Uses matplotlib 3D plotting + ffmpeg — no GPU or display server needed.
"""

import tempfile
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")  # headless backend

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from matplotlib.patches import FancyBboxPatch
from scipy.interpolate import CubicSpline


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
    center = coords.mean(axis=0)
    coords -= center

    return {
        "coords": coords,
        "residue_numbers": residue_numbers,
        "bfactors": np.array(bfactors),
        "residue_names": residue_names,
        "center": center,
    }


def _smooth_backbone(coords: np.ndarray, factor: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Cubic spline interpolation to smooth the backbone into a ribbon-like curve.

    Returns:
        smooth_coords: (N*factor, 3) smoothed coordinates.
        param_t: (N*factor,) parameter values mapping back to original residue indices.
    """
    n = len(coords)
    if n < 4:
        return coords, np.arange(n, dtype=float)

    t_orig = np.arange(n, dtype=float)
    t_smooth = np.linspace(0, n - 1, n * factor)

    smooth = np.zeros((len(t_smooth), 3))
    for dim in range(3):
        cs = CubicSpline(t_orig, coords[:, dim], bc_type="natural")
        smooth[:, dim] = cs(t_smooth)

    return smooth, t_smooth


def _get_risk_color(risk_level: str) -> str:
    return {"HIGH": "#ef4444", "MEDIUM": "#f59e0b", "LOW": "#22c55e"}.get(risk_level, "#94a3b8")


def _rotation_matrix(angle_deg: float, axis: str = "y") -> np.ndarray:
    theta = np.radians(angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    elif axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    else:
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _color_backbone_smooth(
    backbone: dict,
    pocket_residues: list[int],
    danger_residues: list[int],
    param_t: np.ndarray,
) -> np.ndarray:
    """Assign colors to each interpolated point on the smooth backbone."""
    n_orig = len(backbone["residue_numbers"])
    n_smooth = len(param_t)
    colors = np.zeros((n_smooth, 4))

    pocket_set = set(pocket_residues)
    danger_set = set(danger_residues)

    for i, t in enumerate(param_t):
        orig_idx = int(round(t))
        orig_idx = min(orig_idx, n_orig - 1)
        resnum = backbone["residue_numbers"][orig_idx]
        plddt = backbone["bfactors"][orig_idx]

        if resnum in danger_set:
            colors[i] = [0.94, 0.27, 0.27, 1.0]  # red
        elif resnum in pocket_set:
            colors[i] = [0.96, 0.62, 0.04, 1.0]  # orange
        else:
            t_plddt = np.clip((plddt - 50) / 50, 0, 1)
            r = 0.35 - 0.15 * t_plddt
            g = 0.50 - 0.05 * t_plddt
            b = 0.75 + 0.25 * t_plddt
            colors[i] = [r, g, b, 0.95]

    return colors


def _draw_ribbon(ax, rotated: np.ndarray, colors: np.ndarray, base_width: float = 3.5):
    """Draw the backbone as a multi-layered ribbon with glow effect."""
    n = len(rotated)
    if n < 2:
        return

    # Layer definitions: (width_multiplier, alpha_multiplier)
    layers = [
        (base_width * 2.2, 0.08),  # outer glow
        (base_width * 1.4, 0.20),  # mid glow
        (base_width, 0.85),        # core
        (base_width * 0.4, 0.50),  # bright center
    ]

    for width, alpha_mult in layers:
        i = 0
        while i < n - 1:
            # Batch consecutive points with similar color
            j = i + 1
            while j < n and np.allclose(colors[i, :3], colors[j, :3], atol=0.05):
                j += 1
            j = min(j, n)

            seg = rotated[i:j]
            c = colors[i]
            seg_alpha = float(c[3]) * alpha_mult

            if alpha_mult > 0.5:
                ax.plot(
                    seg[:, 0], seg[:, 1], seg[:, 2],
                    color=(float(c[0]), float(c[1]), float(c[2]), seg_alpha),
                    linewidth=width, solid_capstyle="round", solid_joinstyle="round",
                )
            else:
                # Glow layers: lighten the color
                glow_r = min(1.0, float(c[0]) * 0.5 + 0.5)
                glow_g = min(1.0, float(c[1]) * 0.5 + 0.5)
                glow_b = min(1.0, float(c[2]) * 0.5 + 0.5)
                ax.plot(
                    seg[:, 0], seg[:, 1], seg[:, 2],
                    color=(glow_r, glow_g, glow_b, seg_alpha),
                    linewidth=width, solid_capstyle="round", solid_joinstyle="round",
                )
            i = j


def _draw_residue_markers(
    ax, rotated: np.ndarray, backbone: dict,
    danger_residues: list[int], pocket_residues: list[int],
    t_sec: float,
):
    """Draw markers for danger and pocket residues."""
    danger_set = set(danger_residues)
    pocket_set = set(pocket_residues)
    resnums = backbone["residue_numbers"]

    # Pocket residues — orange spheres
    pocket_mask = np.array([r in pocket_set for r in resnums])
    if pocket_mask.any():
        pocket_pts = rotated[pocket_mask]
        ax.scatter(
            pocket_pts[:, 0], pocket_pts[:, 1], pocket_pts[:, 2],
            c="#f59e0b", s=60, alpha=0.7, edgecolors="#fcd34d",
            linewidth=0.8, depthshade=True, zorder=5,
        )

    # Danger residues — red spheres with pulsing glow
    danger_mask = np.array([r in danger_set for r in resnums])
    if danger_mask.any():
        danger_pts = rotated[danger_mask]
        if 7.0 <= t_sec < 9.5:
            pulse = 0.5 + 0.5 * np.sin(t_sec * 8)
            s_inner = 100 + 60 * pulse
            s_glow = 250 + 100 * pulse
        else:
            s_inner = 70
            s_glow = 160

        # Outer glow
        ax.scatter(
            danger_pts[:, 0], danger_pts[:, 1], danger_pts[:, 2],
            c="#ff6b6b", s=s_glow, alpha=0.12, edgecolors="none",
            depthshade=False, zorder=4,
        )
        # Inner sphere
        ax.scatter(
            danger_pts[:, 0], danger_pts[:, 1], danger_pts[:, 2],
            c="#ef4444", s=s_inner, alpha=0.85, edgecolors="#fca5a5",
            linewidth=0.8, depthshade=True, zorder=6,
        )


def _draw_stats_panel(fig, video_data: ProteinVideoData, alpha: float = 1.0):
    """Draw a translucent stats overlay panel."""
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

    ax_s = fig.add_axes([0.02, 0.02, 0.35, 0.55], facecolor="none")
    ax_s.set_xlim(0, 1)
    ax_s.set_ylim(0, 1)
    ax_s.axis("off")

    bg = FancyBboxPatch(
        (0.02, 0.02), 0.96, 0.96,
        boxstyle="round,pad=0.03",
        facecolor=(0.05, 0.05, 0.12, 0.85 * alpha),
        edgecolor=(0.3, 0.3, 0.4, 0.6 * alpha),
        linewidth=1.5,
    )
    ax_s.add_patch(bg)

    ta = alpha
    y = 0.90

    ax_s.text(0.5, y, "SCREENING RESULTS", ha="center", va="top",
              fontsize=9, fontweight="bold", color=(0.7, 0.8, 1.0, ta),
              fontfamily="monospace")
    y -= 0.11

    ax_s.text(0.08, y, "Risk Score", va="top", fontsize=8,
              color=(0.7, 0.7, 0.8, ta), fontfamily="monospace")
    ax_s.text(0.92, y, f"{video_data.risk_score:.3f}", va="top", ha="right",
              fontsize=14, fontweight="bold", color=risk_color,
              fontfamily="monospace", alpha=ta)
    y -= 0.09
    ax_s.text(0.08, y, f"{video_data.risk_level}", va="top", fontsize=8,
              fontweight="bold", color=risk_color,
              fontfamily="monospace", alpha=ta)
    y -= 0.10

    ax_s.plot([0.08, 0.92], [y + 0.03, y + 0.03],
              color=(0.3, 0.3, 0.5, 0.5 * alpha), linewidth=0.8)

    ax_s.text(0.08, y, "Top Match", va="top", fontsize=7,
              color=(0.6, 0.6, 0.7, ta), fontfamily="monospace")
    y -= 0.08
    display_name = match_name[:22] + "..." if len(match_name) > 25 else match_name
    ax_s.text(0.08, y, display_name, va="top", fontsize=7.5,
              fontweight="bold", color=(0.9, 0.9, 1.0, ta),
              fontfamily="monospace")
    y -= 0.07
    if match_org:
        display_org = match_org[:25] + "..." if len(match_org) > 28 else match_org
        ax_s.text(0.08, y, display_org, va="top", fontsize=6.5,
                  fontstyle="italic", color=(0.5, 0.5, 0.6, ta),
                  fontfamily="monospace")
        y -= 0.09

    bars = [("Embedding", emb_sim)]
    if struct_sim is not None:
        bars.append(("Structure", struct_sim))
    bars.append(("Function", func_overlap))

    for label, val in bars:
        ax_s.text(0.08, y, label, va="top", fontsize=6.5,
                  color=(0.6, 0.6, 0.7, ta), fontfamily="monospace")
        ax_s.text(0.92, y, f"{val:.3f}", va="top", ha="right", fontsize=6.5,
                  color=(0.9, 0.9, 1.0, ta), fontfamily="monospace")
        y -= 0.06
        bar_y = y + 0.02
        bar_h = 0.025
        bg_bar = FancyBboxPatch(
            (0.08, bar_y), 0.84, bar_h,
            boxstyle="round,pad=0.005",
            facecolor=(0.15, 0.15, 0.25, 0.6 * alpha),
            edgecolor="none",
        )
        ax_s.add_patch(bg_bar)
        fill_w = max(0.01, val * 0.84)
        bar_color = _get_risk_color("HIGH") if val > 0.7 else ("#f59e0b" if val > 0.4 else "#22c55e")
        fill_bar = FancyBboxPatch(
            (0.08, bar_y), fill_w, bar_h,
            boxstyle="round,pad=0.005",
            facecolor=(*matplotlib.colors.to_rgb(bar_color), 0.8 * alpha),
            edgecolor="none",
        )
        ax_s.add_patch(fill_bar)
        y -= 0.07

    ax_s.text(0.08, y, f"Length: {video_data.sequence_length} aa", va="top",
              fontsize=6.5, color=(0.5, 0.5, 0.6, ta), fontfamily="monospace")


def _draw_title_card(fig, video_data: ProteinVideoData, alpha: float = 1.0):
    if alpha <= 0:
        return

    risk_color = _get_risk_color(video_data.risk_level)

    ax_t = fig.add_axes([0.0, 0.0, 1.0, 1.0], facecolor="none")
    ax_t.set_xlim(0, 1)
    ax_t.set_ylim(0, 1)
    ax_t.axis("off")

    bg = FancyBboxPatch(
        (0.15, 0.25), 0.70, 0.50,
        boxstyle="round,pad=0.04",
        facecolor=(0.05, 0.05, 0.12, 0.92 * alpha),
        edgecolor=(0.3, 0.3, 0.5, 0.6 * alpha),
        linewidth=2,
    )
    ax_t.add_patch(bg)

    ax_t.text(0.5, 0.68, "BIOSCREEN", ha="center", va="center",
              fontsize=22, fontweight="bold", color=(0.5, 0.7, 1.0, alpha),
              fontfamily="monospace")
    ax_t.text(0.5, 0.60, "Protein Risk Analysis", ha="center", va="center",
              fontsize=12, color=(0.7, 0.7, 0.8, alpha), fontfamily="monospace")

    ax_t.text(0.5, 0.48, f"Sequence: {video_data.sequence_length} amino acids",
              ha="center", va="center", fontsize=9,
              color=(0.6, 0.6, 0.7, alpha), fontfamily="monospace")

    ax_t.text(0.5, 0.38, f"Risk: {video_data.risk_level}", ha="center", va="center",
              fontsize=16, fontweight="bold", color=risk_color,
              fontfamily="monospace", alpha=alpha)

    mode = "Full (Structure + Embedding + Function)"
    ax_t.text(0.5, 0.30, f"Mode: {mode}", ha="center", va="center",
              fontsize=7.5, color=(0.5, 0.5, 0.6, alpha), fontfamily="monospace")


def _draw_danger_label(fig, video_data: ProteinVideoData, alpha: float = 1.0):
    if alpha <= 0 or not video_data.danger_residues:
        return

    ax_l = fig.add_axes([0.55, 0.78, 0.43, 0.18], facecolor="none")
    ax_l.set_xlim(0, 1)
    ax_l.set_ylim(0, 1)
    ax_l.axis("off")

    bg = FancyBboxPatch(
        (0.02, 0.02), 0.96, 0.96,
        boxstyle="round,pad=0.03",
        facecolor=(0.3, 0.05, 0.05, 0.85 * alpha),
        edgecolor=(0.94, 0.27, 0.27, 0.7 * alpha),
        linewidth=1.5,
    )
    ax_l.add_patch(bg)

    ax_l.text(0.5, 0.65, "DANGER ZONE", ha="center", va="center",
              fontsize=10, fontweight="bold", color=(1.0, 0.3, 0.3, alpha),
              fontfamily="monospace")
    ax_l.text(0.5, 0.30, f"{len(video_data.danger_residues)} residues match toxin active site",
              ha="center", va="center", fontsize=7,
              color=(0.9, 0.6, 0.6, alpha), fontfamily="monospace")


def _draw_verdict(fig, video_data: ProteinVideoData, alpha: float = 1.0):
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
    """
    backbone = parse_pdb_backbone(video_data.pdb_string)
    coords = backbone["coords"]

    smooth_coords, param_t = _smooth_backbone(coords, factor=4)
    smooth_colors = _color_backbone_smooth(
        backbone, video_data.pocket_residues, video_data.danger_residues, param_t,
    )

    total_frames = int(fps * duration)

    danger_set = set(video_data.danger_residues)
    danger_mask = np.array([r in danger_set for r in backbone["residue_numbers"]])
    if danger_mask.any():
        danger_center = coords[danger_mask].mean(axis=0)
    else:
        danger_center = np.zeros(3)

    fig = plt.figure(figsize=(10, 6), dpi=150, facecolor="#080812")
    max_extent = np.abs(coords).max() * 1.3

    def animate(frame_idx):
        fig.clf()
        ax_3d = fig.add_axes([0.02, 0.02, 0.96, 0.96], projection="3d", facecolor="#080812")
        fig.patch.set_facecolor("#080812")

        t = frame_idx / total_frames
        t_sec = t * duration

        azim = -60 + t * 360
        elev = 20 + 8 * np.sin(t * np.pi * 2)

        zoom = max_extent
        use_shifted = False

        if 7.0 <= t_sec < 9.5 and danger_mask.any():
            zoom_t = (t_sec - 7.0) / 2.5
            zoom_t = 3 * zoom_t**2 - 2 * zoom_t**3  # smooth ease
            zoom = max_extent * (1.0 - 0.55 * zoom_t)
            view_center = danger_center * zoom_t
            shifted_smooth = smooth_coords - view_center
            shifted_orig = coords - view_center
            use_shifted = True

        rot = _rotation_matrix(azim, "y") @ _rotation_matrix(elev, "x")

        if use_shifted:
            rotated_smooth = (rot @ shifted_smooth.T).T
            rotated_orig = (rot @ shifted_orig.T).T
        else:
            rotated_smooth = (rot @ smooth_coords.T).T
            rotated_orig = (rot @ coords.T).T

        _draw_ribbon(ax_3d, rotated_smooth, smooth_colors, base_width=3.5)

        # Draw residue markers on original CA positions
        _draw_residue_markers(
            ax_3d, rotated_orig, backbone,
            video_data.danger_residues, video_data.pocket_residues, t_sec,
        )

        ax_3d.set_xlim(-zoom, zoom)
        ax_3d.set_ylim(-zoom, zoom)
        ax_3d.set_zlim(-zoom, zoom)
        ax_3d.axis("off")
        ax_3d.set_box_aspect([1, 1, 1])

        # --- Timeline overlays ---

        if t_sec < 2.0:
            if t_sec < 0.5:
                a = t_sec / 0.5
            elif t_sec > 1.5:
                a = (2.0 - t_sec) / 0.5
            else:
                a = 1.0
            _draw_title_card(fig, video_data, alpha=a)

        if 2.0 <= t_sec <= 9.5:
            if t_sec < 2.5:
                a = (t_sec - 2.0) / 0.5
            elif t_sec > 9.0:
                a = (9.5 - t_sec) / 0.5
            else:
                a = 1.0
            _draw_stats_panel(fig, video_data, alpha=a)

        if 7.0 <= t_sec < 9.5 and video_data.danger_residues:
            if t_sec < 7.5:
                a = (t_sec - 7.0) / 0.5
            elif t_sec > 9.0:
                a = (9.5 - t_sec) / 0.5
            else:
                a = 1.0
            _draw_danger_label(fig, video_data, alpha=a)

        if t_sec >= 9.5:
            a = min(1.0, (t_sec - 9.5) / 0.5)
            _draw_verdict(fig, video_data, alpha=a)

        fig.text(0.98, 0.02, "BioScreen", ha="right", va="bottom",
                 fontsize=7, color=(0.3, 0.3, 0.4, 0.6), fontfamily="monospace")

        return []

    anim = animation.FuncAnimation(
        fig, animate, frames=total_frames, interval=1000 / fps, blit=False,
    )

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
