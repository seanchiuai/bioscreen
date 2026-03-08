import py3Dmol
import streamlit.components.v1 as components


def _aligned_residue_set(aligned_regions: list[list[int]]) -> list[int]:
    """Expand [start, end] pairs into a deduplicated sorted list of residue indices."""
    residues: set[int] = set()
    for region in aligned_regions:
        if len(region) == 2:
            residues.update(range(region[0], region[1] + 1))
    return sorted(residues)


def render_protein_3d(
    pdb_string: str,
    pocket_residues: list[int],
    danger_residues: list[int],
    aligned_regions: list[list[int]] | None = None,
    view_style: str = "Cartoon",
    color_mode: str = "Default",
    overlay_pdb: str | None = None,
    overlay_name: str = "",
    width: int = 600,
    height: int = 480,
) -> None:
    """Render an interactive 3D protein viewer using py3Dmol.

    Args:
        pdb_string: PDB format string from ESMFold.
        pocket_residues: Residue indices for active site pockets (orange).
        danger_residues: Residue indices matching toxin active sites (red).
        aligned_regions: Regions structurally aligned to toxin, as [start, end] pairs.
        view_style: One of "Cartoon", "Surface", "Stick".
        color_mode: "Default", "pLDDT", or "Risk Layers".
        overlay_pdb: Optional aligned toxin PDB string for superposition overlay.
        overlay_name: Name of the overlaid toxin (for display).
        width: Viewer width in pixels.
        height: Viewer height in pixels.
    """
    view = py3Dmol.view(width=width, height=height)
    view.addModel(pdb_string, "pdb")

    if color_mode == "Risk Layers":
        # Layer 1 (global): gray base, yellow for structurally aligned regions
        if view_style == "Cartoon":
            view.setStyle({"model": 0}, {"cartoon": {"color": "#b0b0b0"}})
        elif view_style == "Surface":
            view.setStyle({"model": 0}, {"cartoon": {"color": "#b0b0b0", "opacity": 0.5}})
            view.addSurface(py3Dmol.VDW, {"opacity": 0.5, "color": "#b0b0b0"}, {"model": 0})
        elif view_style == "Stick":
            view.setStyle({"model": 0}, {"stick": {"color": "#b0b0b0"}})

        # Highlight aligned backbone regions in yellow
        aligned_res = _aligned_residue_set(aligned_regions or [])
        if aligned_res:
            if view_style == "Cartoon":
                view.addStyle({"model": 0, "resi": aligned_res}, {"cartoon": {"color": "#fbbf24"}})
            elif view_style == "Surface":
                view.addStyle({"model": 0, "resi": aligned_res}, {"cartoon": {"color": "#fbbf24", "opacity": 0.5}})
                view.addSurface(py3Dmol.VDW, {"opacity": 0.5, "color": "#fbbf24"}, {"model": 0, "resi": aligned_res})
            elif view_style == "Stick":
                view.addStyle({"model": 0, "resi": aligned_res}, {"stick": {"color": "#fbbf24"}})

        # Layer 2 (local): pocket residues in orange, danger residues in red
        if pocket_residues:
            view.addStyle(
                {"model": 0, "resi": pocket_residues},
                {"stick": {"color": "orange", "radius": 0.2}},
            )
        if danger_residues:
            view.addStyle(
                {"model": 0, "resi": danger_residues},
                {"stick": {"color": "red", "radius": 0.3}},
            )
            view.addSurface(
                py3Dmol.VDW,
                {"opacity": 0.3, "color": "red"},
                {"model": 0, "resi": danger_residues},
            )
    else:
        # Original color modes (Default / pLDDT)
        if color_mode == "pLDDT" and not overlay_pdb:
            color_spec = {
                "prop": "b",
                "gradient": "roygb",
                "min": 50,
                "max": 100,
            }
        else:
            color_spec = "lightblue"

        # Style the query protein (model 0)
        if view_style == "Cartoon":
            if isinstance(color_spec, dict):
                view.setStyle({"model": 0}, {"cartoon": {"colorscheme": color_spec}})
            else:
                view.setStyle({"model": 0}, {"cartoon": {"color": color_spec}})
        elif view_style == "Surface":
            if isinstance(color_spec, dict):
                view.setStyle({"model": 0}, {"cartoon": {"colorscheme": color_spec, "opacity": 0.5}})
                view.addSurface(
                    py3Dmol.VDW,
                    {"opacity": 0.7, "colorscheme": color_spec},
                    {"model": 0},
                )
            else:
                view.setStyle({"model": 0}, {"cartoon": {"color": color_spec, "opacity": 0.5}})
                view.addSurface(
                    py3Dmol.VDW,
                    {"opacity": 0.7, "color": color_spec},
                    {"model": 0},
                )
        elif view_style == "Stick":
            if isinstance(color_spec, dict):
                view.setStyle({"model": 0}, {"stick": {"colorscheme": color_spec}})
            else:
                view.setStyle({"model": 0}, {"stick": {"color": color_spec}})

        # Highlight pocket residues in orange (stick representation)
        if pocket_residues:
            view.addStyle(
                {"model": 0, "resi": pocket_residues},
                {"stick": {"color": "orange", "radius": 0.2}},
            )

        # Highlight danger residues in red (thick stick + transparent surface)
        if danger_residues:
            view.addStyle(
                {"model": 0, "resi": danger_residues},
                {"stick": {"color": "red", "radius": 0.3}},
            )
            view.addSurface(
                py3Dmol.VDW,
                {"opacity": 0.3, "color": "red"},
                {"model": 0, "resi": danger_residues},
            )

    # Overlay toxin structure (model 1) — semi-transparent red
    if overlay_pdb:
        view.addModel(overlay_pdb, "pdb")
        view.setStyle(
            {"model": 1},
            {"cartoon": {"color": "#e74c3c", "opacity": 0.5}},
        )
        view.addSurface(
            py3Dmol.VDW,
            {"opacity": 0.2, "color": "#e74c3c"},
            {"model": 1},
        )

    view.zoomTo()
    view.spin(False)

    # Render into Streamlit via HTML iframe
    html = view._make_html()
    components.html(html, width=width, height=height, scrolling=False)
