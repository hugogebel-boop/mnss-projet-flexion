# Projet 5 — Flexion d'une poutre 2D

CIVIL-321 (Modélisation Numérique des Solides et Structures, EPFL).

Implémentation FEM 2D des éléments triangulaires linéaire (T3) et quadratique
iso-paramétrique (T6), validée par patch test, appliquée à une poutre
encastrée-appuyée sous charge répartie. Comparaison T3 / T6 / Euler-Bernoulli,
diagnostic du shear locking.

## Structure

```
07_Projet/
├── Projet5_Flexion.ipynb   notebook principal, paramètres en haut
├── plot.py, utils.py       helpers fournis
├── fem_utils.py            briques FEM réutilisables
├── Template.ipynb          squelette d'origine (référence)
└── Rapport/
    ├── rapport.tex         rapport 5 pages (LaTeX)
    ├── rapport.pdf         build courant
    ├── extract_figures.py  extraction PNG depuis le .ipynb
    ├── regenerate_figures.py  variantes propres pour le rapport
    ├── build.bat           wrapper pdflatex
    └── figures/            PNG inclus dans le rapport
```

## Exécuter le notebook

Pré-requis : Python 3.13, environnement virtuel à `Notebooks/.venv` avec les
dépendances de `Notebooks/requirements.txt` (numpy, scipy, matplotlib,
ipykernel...).

Le notebook embarque ses paramètres globaux en tête (cellule `MAT`, `GEOM`,
`MESH`, `LOAD`...). Il bascule entre T3 et T6 par `MESH["order"]` (1 ou 2).

```bash
# kernel pré-enregistré pour la venv
"Notebooks/.venv/Scripts/python.exe" -m jupyter nbconvert \
    --to notebook --execute --inplace \
    --ExecutePreprocessor.kernel_name=python3 \
    --ExecutePreprocessor.timeout=900 \
    "Notebooks/07_Projet/Projet5_Flexion.ipynb"
```

Sous VS Code, sélectionner le kernel **Python (MNSS Projet5)** ou pointer
vers `Notebooks/.venv/Scripts/python.exe`. Temps d'exécution complet
~6 min (la convergence inclut une référence T6 ultra-fine 320×64).

## Compiler le rapport

Pré-requis : MiKTeX (ou TeX Live), Pygments (`pip install Pygments`) pour
le paquet `minted`. Auto-install MiKTeX activable par
`initexmf --set-config-value '[MPM]AutoInstall=1'`.

```bash
cd Rapport
./build.bat              # ou pdflatex -shell-escape rapport.tex (×2)
```

PDF de sortie : `Rapport/rapport.pdf` (5 pages).

## Phases livrées

- Phase 2 — fonctions de forme T6, dérivées, quadrature de Gauss (1/3/7 pts).
- Phase 3 — patch test traction + cisaillement, géométrie distordue, erreur
  ~1e-19.
- Phase 4 — poutre encastrée-appuyée, comparaison Euler-Bernoulli.
- Phase 5 — convergence h log-log, étude L/h, analyse modale.
- Phase 6 — diagnostic shear locking, évaluation T6, alternatives.

Détail dans le rapport et dans la TODO list à la fin du notebook.
