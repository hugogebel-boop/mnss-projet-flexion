"""Extrait les figures PNG du notebook (sans le ré-exécuter)."""
import base64
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
NB   = ROOT / "Projet5_Flexion.ipynb"
FIGS = Path(__file__).parent / "figures"
FIGS.mkdir(exist_ok=True)

# (cell_id, output_index) -> filename
WANTED = {
    ("030d653a", 1): "patch_meshes.png",        # Phase 3 : T3 + T6 fig.10
    ("1cfc9ccf", 1): "beam_deflection.png",     # Phase 4 : profil v(x) FEM vs EB
    ("c597c6b5", 0): "beam_deformed.png",       # Phase 4 : déformée + uy
    ("c597c6b5", 1): "sigma_field.png",         # Phase 4 : carte σxx
    ("c597c6b5", 2): "sigma_profile.png",       # Phase 4 : profil σxx fibres extrêmes
    ("b1d644a8", 9): "convergence_h.png",       # Phase 5a : log-log T3/T6
    ("2699d0a1", 7): "err_vs_slenderness.png",  # Phase 5b : err vs L/h
    ("232bbb3e", 2): "modes.png",               # Phase 5c : 3 premiers modes
    ("c5b11c3a", 4): "locking_demo.png",        # Phase 6  : T3 ny variable
}

nb = json.loads(NB.read_text(encoding="utf8"))
for cell in nb["cells"]:
    cid = cell.get("id")
    for (cid_match, idx), fname in WANTED.items():
        if cid_match != cid:
            continue
        outputs = cell.get("outputs", [])
        if idx >= len(outputs):
            print(f"  WARN ({cid}, {idx}): only {len(outputs)} outputs")
            continue
        data = outputs[idx].get("data", {}).get("image/png")
        if not data:
            print(f"  WARN ({cid}, {idx}): no image/png")
            continue
        (FIGS / fname).write_bytes(base64.b64decode(data))
        print(f"  -> {fname}")
print(f"Done. {len(list(FIGS.glob('*.png')))} PNG dans {FIGS}")
