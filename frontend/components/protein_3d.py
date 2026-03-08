import json
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
    """Render an interactive 3D protein viewer using 3Dmol.js (inline embed).

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
    aligned_res = _aligned_residue_set(aligned_regions or [])

    # Escape PDB strings for embedding in JS
    pdb_json = json.dumps(pdb_string)
    overlay_json = json.dumps(overlay_pdb) if overlay_pdb else "null"
    pocket_json = json.dumps(pocket_residues)
    danger_json = json.dumps(danger_residues)
    aligned_json = json.dumps(aligned_res)

    # Map view style to 3Dmol representation
    style_map = {"Cartoon": "cartoon", "Surface": "surface", "Stick": "stick"}
    rep = style_map.get(view_style, "cartoon")

    html = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  body {{ margin: 0; padding: 0; background: white; }}
  #viewer {{ width: {width}px; height: {height}px; position: relative; }}
</style>
</head>
<body>
<div id="viewer"></div>
<script src="https://3dmol.org/build/3Dmol-min.js"></script>
<script>
(function() {{
  var pdbData = {pdb_json};
  var overlayData = {overlay_json};
  var pocketRes = {pocket_json};
  var dangerRes = {danger_json};
  var alignedRes = {aligned_json};
  var colorMode = {json.dumps(color_mode)};
  var rep = {json.dumps(rep)};

  var viewer = $3Dmol.createViewer("viewer", {{
    backgroundColor: "white",
  }});

  viewer.addModel(pdbData, "pdb");

  if (colorMode === "Risk Layers") {{
    // Base: gray
    var baseStyle = {{}};
    baseStyle[rep] = {{color: "#b0b0b0"}};
    viewer.setStyle({{model: 0}}, baseStyle);

    // Yellow: structurally aligned regions
    if (alignedRes.length > 0) {{
      var alignStyle = {{}};
      alignStyle[rep] = {{color: "#fbbf24"}};
      viewer.addStyle({{model: 0, resi: alignedRes}}, alignStyle);
    }}
  }} else if (colorMode === "pLDDT") {{
    if (rep === "cartoon") {{
      viewer.setStyle({{model: 0}}, {{cartoon: {{colorscheme: {{prop: "b", gradient: "roygb", min: 50, max: 100}}}}}});
    }} else {{
      var s = {{}};
      s[rep] = {{colorscheme: {{prop: "b", gradient: "roygb", min: 50, max: 100}}}};
      viewer.setStyle({{model: 0}}, s);
    }}
  }} else {{
    // Default: lightblue
    if (rep === "surface") {{
      viewer.setStyle({{model: 0}}, {{cartoon: {{color: "lightblue", opacity: 0.5}}}});
      viewer.addSurface($3Dmol.SurfaceType.VDW, {{opacity: 0.7, color: "lightblue"}}, {{model: 0}});
    }} else {{
      var s = {{}};
      s[rep] = {{color: "lightblue"}};
      viewer.setStyle({{model: 0}}, s);
    }}
  }}

  // Pocket residues: orange sticks
  if (pocketRes.length > 0) {{
    viewer.addStyle({{model: 0, resi: pocketRes}}, {{stick: {{color: "orange", radius: 0.2}}}});
  }}

  // Danger residues: red sticks + transparent surface
  if (dangerRes.length > 0) {{
    viewer.addStyle({{model: 0, resi: dangerRes}}, {{stick: {{color: "red", radius: 0.3}}}});
    viewer.addSurface($3Dmol.SurfaceType.VDW, {{opacity: 0.3, color: "red"}}, {{model: 0, resi: dangerRes}});
  }}

  // Overlay toxin: semi-transparent red cartoon
  if (overlayData) {{
    viewer.addModel(overlayData, "pdb");
    viewer.setStyle({{model: 1}}, {{cartoon: {{color: "#e74c3c", opacity: 0.5}}}});
    viewer.addSurface($3Dmol.SurfaceType.VDW, {{opacity: 0.2, color: "#e74c3c"}}, {{model: 1}});
  }}

  viewer.zoomTo();
  viewer.render();
  viewer.spin(false);
}})();
</script>
</body>
</html>
"""

    components.html(html, width=width, height=height + 10, scrolling=False)
