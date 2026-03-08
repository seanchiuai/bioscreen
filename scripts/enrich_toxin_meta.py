#!/usr/bin/env python3
"""Enrich toxin_meta.json with human-readable descriptions, targets, and mechanisms.

Reads the existing data/toxin_meta.json and adds:
  - danger_description: 1-line human-readable explanation of what the toxin does
  - biological_target: what the toxin targets (receptor, channel, ribosome, etc.)
  - mechanism: how it causes harm (blocker, inhibitor, pore-forming, etc.)
  - toxin_type: refined classification (replaces the coarse ion_channel_toxin label)

All classification is done via GO term matching, organism heuristics, and protein
name keyword matching.  No external API calls.

Usage:
    python scripts/enrich_toxin_meta.py
    python scripts/enrich_toxin_meta.py --dry-run   # preview without writing
"""

import argparse
import json
import re
import sys
from pathlib import Path

META_PATH = Path(__file__).parent.parent / "data" / "toxin_meta.json"

# ── GO term → classification rules ──────────────────────────────────────────

# Each rule: (go_term_substring, toxin_type, biological_target, mechanism, danger_description)
GO_RULES: list[tuple[str, str, str, str, str]] = [
    # Specific channel/receptor targets (most specific first)
    (
        "potassium channel inhibitor",
        "potassium_channel_toxin",
        "Voltage-gated potassium channels (Kv)",
        "Ion channel blocker",
        "Blocks potassium channels, disrupting nerve and muscle cell repolarization",
    ),
    (
        "potassium channel regulator",
        "potassium_channel_toxin",
        "Voltage-gated potassium channels (Kv)",
        "Ion channel modulator",
        "Modulates potassium channel gating, altering neuronal excitability",
    ),
    (
        "sodium channel inhibitor",
        "sodium_channel_toxin",
        "Voltage-gated sodium channels (Nav)",
        "Ion channel blocker",
        "Blocks sodium channels, preventing nerve impulse propagation",
    ),
    (
        "sodium channel regulator",
        "sodium_channel_toxin",
        "Voltage-gated sodium channels (Nav)",
        "Ion channel modulator",
        "Modulates sodium channel gating, causing abnormal nerve firing",
    ),
    (
        "calcium channel inhibitor",
        "calcium_channel_toxin",
        "Voltage-gated calcium channels (Cav)",
        "Ion channel blocker",
        "Blocks calcium channels, inhibiting neurotransmitter release",
    ),
    (
        "calcium channel regulator",
        "calcium_channel_toxin",
        "Voltage-gated calcium channels (Cav)",
        "Ion channel modulator",
        "Modulates calcium channel function, disrupting synaptic transmission",
    ),
    (
        "chloride channel regulator",
        "chloride_channel_toxin",
        "Chloride channels (ClC)",
        "Ion channel modulator",
        "Modulates chloride channels, altering inhibitory signaling",
    ),
    (
        "acetylcholine receptor inhibitor",
        "nicotinic_receptor_toxin",
        "Nicotinic acetylcholine receptors (nAChR)",
        "Receptor antagonist",
        "Blocks nicotinic acetylcholine receptors, causing neuromuscular paralysis",
    ),
    (
        "acetylcholine receptor activator",
        "nicotinic_receptor_toxin",
        "Nicotinic acetylcholine receptors (nAChR)",
        "Receptor agonist",
        "Activates nicotinic receptors, causing uncontrolled muscle contraction",
    ),
    # Enzymatic toxins
    (
        "phospholipase A2",
        "phospholipase_toxin",
        "Cell membrane phospholipids",
        "Enzymatic membrane disruption",
        "Hydrolyzes cell membrane phospholipids, causing tissue destruction and inflammation",
    ),
    (
        "phospholipid binding",
        "phospholipase_toxin",
        "Cell membrane phospholipids",
        "Membrane disruption",
        "Binds and disrupts cell membrane phospholipids",
    ),
    (
        "serine-type endopeptidase inhibitor",
        "protease_inhibitor_toxin",
        "Serine proteases (trypsin, chymotrypsin)",
        "Protease inhibitor",
        "Inhibits serine proteases, disrupting blood coagulation and immune response",
    ),
    (
        "serine-type endopeptidase activity",
        "serine_protease_toxin",
        "Protein substrates",
        "Proteolytic enzyme",
        "Cleaves proteins via serine protease activity, causing tissue damage",
    ),
    (
        "rRNA N-glycosylase",
        "ribosome_inactivating_toxin",
        "28S ribosomal RNA",
        "Ribosome inactivation",
        "Depurinates 28S rRNA, irreversibly shutting down protein synthesis in target cells",
    ),
    # General ion channel
    (
        "ion channel inhibitor",
        "ion_channel_toxin",
        "Ion channels",
        "Ion channel blocker",
        "Blocks ion channels, disrupting nerve and muscle signal transmission",
    ),
    (
        "ion channel regulator",
        "ion_channel_toxin",
        "Ion channels",
        "Ion channel modulator",
        "Modulates ion channel gating, altering cellular excitability",
    ),
]

# ── Name keyword → classification rules ─────────────────────────────────────

# Fallback rules based on protein name keywords (checked if no GO rule matched)
NAME_RULES: list[tuple[str, str, str, str, str]] = [
    # Conotoxins by Greek prefix
    (
        r"alpha-conotoxin",
        "nicotinic_receptor_toxin",
        "Nicotinic acetylcholine receptors (nAChR)",
        "Receptor antagonist",
        "Blocks nicotinic acetylcholine receptors at neuromuscular junctions",
    ),
    (
        r"omega-conotoxin",
        "calcium_channel_toxin",
        "Voltage-gated calcium channels (Cav)",
        "Ion channel blocker",
        "Blocks N-type calcium channels, inhibiting neurotransmitter release",
    ),
    (
        r"mu-conotoxin",
        "sodium_channel_toxin",
        "Voltage-gated sodium channels (Nav)",
        "Ion channel blocker",
        "Blocks skeletal muscle sodium channels, causing paralysis",
    ),
    (
        r"delta-conotoxin",
        "sodium_channel_toxin",
        "Voltage-gated sodium channels (Nav)",
        "Ion channel modulator",
        "Delays sodium channel inactivation, causing prolonged nerve firing",
    ),
    (
        r"kappa-conotoxin",
        "potassium_channel_toxin",
        "Voltage-gated potassium channels (Kv)",
        "Ion channel blocker",
        "Blocks potassium channels, affecting nerve repolarization",
    ),
    # Scorpion toxin prefixes
    (
        r"alpha-ktx",
        "potassium_channel_toxin",
        "Voltage-gated potassium channels (Kv)",
        "Ion channel blocker (pore plugging)",
        "Plugs the pore of potassium channels, blocking ion flow",
    ),
    (
        r"beta-ktx",
        "potassium_channel_toxin",
        "Voltage-gated potassium channels (Kv)",
        "Ion channel blocker",
        "Blocks potassium channels with distinct binding mode",
    ),
    # Snake toxin families
    (
        r"three-finger|3ftx",
        "three_finger_toxin",
        "Nicotinic acetylcholine receptors (nAChR)",
        "Receptor antagonist",
        "Three-finger toxin that blocks nicotinic receptors, causing neuromuscular paralysis",
    ),
    (
        r"irditoxin",
        "three_finger_toxin",
        "Nicotinic acetylcholine receptors (nAChR)",
        "Receptor antagonist (dimeric)",
        "Dimeric three-finger toxin that blocks nicotinic acetylcholine receptors",
    ),
    (
        r"phospholipase",
        "phospholipase_toxin",
        "Cell membrane phospholipids",
        "Enzymatic membrane disruption",
        "Hydrolyzes membrane phospholipids, causing hemolysis and tissue necrosis",
    ),
    (
        r"metalloprotease|metalloproteinase",
        "metalloprotease_toxin",
        "Extracellular matrix proteins",
        "Proteolytic enzyme (zinc-dependent)",
        "Zinc metalloprotease that degrades extracellular matrix, causing hemorrhage",
    ),
    # Specific named toxins
    (
        r"ricin",
        "ribosome_inactivating_toxin",
        "28S ribosomal RNA",
        "Ribosome inactivation (RIP type II)",
        "Depurinates ribosomal RNA, halting all protein synthesis — lethal at microgram doses",
    ),
    (
        r"diphtheria",
        "ADP_ribosyltransferase",
        "Elongation factor 2 (EF-2)",
        "ADP-ribosylation",
        "ADP-ribosylates EF-2, irreversibly blocking protein synthesis in human cells",
    ),
    (
        r"hemolysin|haemolysin|alpha-hemolysin|cytolysin",
        "pore_forming_toxin",
        "Cell membranes",
        "Pore formation",
        "Forms transmembrane pores, causing cell lysis through osmotic imbalance",
    ),
    (
        r"neurotoxin",
        "neurotoxin",
        "Nervous system targets",
        "Neurotoxic",
        "Disrupts nervous system function through receptor or channel modulation",
    ),
    (
        r"kunitz",
        "protease_inhibitor_toxin",
        "Serine proteases",
        "Protease inhibitor (Kunitz-type)",
        "Kunitz-type protease inhibitor; some also block ion channels",
    ),
    # Spider toxins
    (
        r"hainantoxin",
        "sodium_channel_toxin",
        "Voltage-gated sodium channels (Nav)",
        "Ion channel blocker",
        "Spider toxin that blocks sodium channels with high selectivity",
    ),
    # Generic fallbacks
    (
        r"beta-toxin|beta.insect",
        "sodium_channel_toxin",
        "Voltage-gated sodium channels (Nav)",
        "Ion channel modulator",
        "Shifts sodium channel activation voltage, causing repetitive nerve firing",
    ),
    (
        r"alpha-toxin|alpha.mammal",
        "sodium_channel_toxin",
        "Voltage-gated sodium channels (Nav)",
        "Ion channel modulator",
        "Slows sodium channel inactivation, prolonging action potentials",
    ),
    (
        r"depressant",
        "sodium_channel_toxin",
        "Voltage-gated sodium channels (Nav)",
        "Ion channel modifier (depressant)",
        "Shifts sodium channel activation, causing flaccid paralysis in insects",
    ),
]

# ── Organism → animal group mapping ─────────────────────────────────────────

ORGANISM_GROUPS: list[tuple[str, str]] = [
    (r"conus\b", "cone snail"),
    (r"buthus|leiurus|androctonus|mesobuthus|tityus|centruroides|hottentotta|babycurus", "scorpion"),
    (r"chilobrachys|cyriopagopus|grammostola|phoneutria|haplopelma|heteroscodra|scolopendra|araneus", "spider/arthropod"),
    (r"naja|bungarus|dendroaspis|boiga|crotalus|bothrops|agkistrodon|trimeresurus|oxyuranus|laticauda|pseudonaja|notechis|daboia", "snake"),
    (r"radianthus|stichodactyla|heteractis", "sea anemone"),
    (r"staphylococcus|clostridium|bacillus|corynebacterium|vibrio|escherichia|bordetella|streptococcus", "bacterium"),
    (r"ricinus|abrus", "plant"),
]


def _get_organism_group(organism: str) -> str:
    lower = organism.lower()
    for pattern, group in ORGANISM_GROUPS:
        if re.search(pattern, lower):
            return group
    return "other"


def enrich_entry(entry: dict) -> dict:
    """Add enrichment fields to a single toxin metadata entry."""
    go_terms_str = " ".join(entry.get("go_terms", [])).lower()
    name_lower = entry.get("name", "").lower()
    organism = entry.get("organism", "")
    organism_group = _get_organism_group(organism)

    toxin_type = entry.get("toxin_type", "unknown")
    biological_target = ""
    mechanism = ""
    danger_description = ""

    # Try GO term rules first (most reliable)
    for go_substr, tt, target, mech, desc in GO_RULES:
        if go_substr.lower() in go_terms_str:
            toxin_type = tt
            biological_target = target
            mechanism = mech
            danger_description = desc
            break

    # Fall back to name keyword rules
    if not danger_description:
        for name_pattern, tt, target, mech, desc in NAME_RULES:
            if re.search(name_pattern, name_lower):
                toxin_type = tt
                biological_target = target
                mechanism = mech
                danger_description = desc
                break

    # Last resort: generic description from existing toxin_type
    if not danger_description:
        if "ion_channel" in toxin_type:
            biological_target = "Ion channels"
            mechanism = "Ion channel modulation"
            danger_description = "Disrupts ion channel function, altering nerve and muscle signaling"
        elif "neurotoxin" in toxin_type:
            biological_target = "Nervous system"
            mechanism = "Neurotoxic"
            danger_description = "Interferes with nervous system signaling"
        elif toxin_type == "toxin" or toxin_type == "unknown":
            biological_target = "Various cellular targets"
            mechanism = "Toxic"
            danger_description = "Toxic protein with potential harmful effects"

    # Add organism context to description if we have it
    if danger_description and organism_group != "other":
        if organism_group in ("bacterium", "plant"):
            danger_description = f"{organism_group.capitalize()} toxin: {danger_description}"
        else:
            danger_description = f"{organism_group.capitalize()} venom: {danger_description}"

    entry["toxin_type"] = toxin_type
    entry["biological_target"] = biological_target
    entry["mechanism"] = mechanism
    entry["danger_description"] = danger_description

    return entry


def main():
    parser = argparse.ArgumentParser(description="Enrich toxin_meta.json with detailed annotations.")
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing")
    parser.add_argument("--meta-path", type=Path, default=META_PATH, help="Path to toxin_meta.json")
    args = parser.parse_args()

    if not args.meta_path.exists():
        print(f"Error: {args.meta_path} not found", file=sys.stderr)
        sys.exit(1)

    with open(args.meta_path) as f:
        data = json.load(f)

    print(f"Loaded {len(data)} entries from {args.meta_path}")

    # Enrich each entry
    for entry in data:
        enrich_entry(entry)

    # Statistics
    from collections import Counter
    types = Counter(d["toxin_type"] for d in data)
    enriched = sum(1 for d in data if d.get("danger_description"))
    with_target = sum(1 for d in data if d.get("biological_target"))

    print(f"\nEnrichment results:")
    print(f"  Entries with danger_description: {enriched}/{len(data)} ({enriched/len(data)*100:.0f}%)")
    print(f"  Entries with biological_target:  {with_target}/{len(data)} ({with_target/len(data)*100:.0f}%)")
    print(f"\nToxin type distribution (refined):")
    for t, c in types.most_common():
        print(f"  {c:5d}  {t}")

    # Show a few examples
    print("\nSample entries:")
    for name_query in ["Irditoxin", "Ricin", "Diphtheria", "Alpha-hemolysin", "Alpha-conotoxin"]:
        matches = [d for d in data if name_query.lower() in d["name"].lower()]
        if matches:
            d = matches[0]
            print(f"\n  {d['name']} ({d['organism']})")
            print(f"    type: {d['toxin_type']}")
            print(f"    target: {d['biological_target']}")
            print(f"    mechanism: {d['mechanism']}")
            print(f"    description: {d['danger_description']}")

    if args.dry_run:
        print("\n[DRY RUN] No changes written.")
    else:
        with open(args.meta_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"\nWritten enriched data to {args.meta_path}")


if __name__ == "__main__":
    main()
