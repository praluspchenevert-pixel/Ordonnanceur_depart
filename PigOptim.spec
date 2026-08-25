# -*- mode: python ; coding: utf-8 -*-
# ============================================================
# PigOptim.spec
# Configuration PyInstaller pour construire l'executable autonome.
# Usage :  pyinstaller PigOptim.spec
# (a la place de la longue commande en ligne, plus fiable car sans
# problemes de guillemets/quoting dans le terminal)
# ============================================================

import sys
from PyInstaller.utils.hooks import collect_all

sys.setrecursionlimit(sys.getrecursionlimit() * 5)

block_cipher = None

# collect_all() renvoie des tuples au format "hook" (source, destination),
# different du format TOC interne (destination, source, type) utilise par
# Analysis une fois construit. Il faut donc les fournir directement au
# constructeur d'Analysis plutot que de les ajouter apres coup a a.datas /
# a.binaries (ce qui provoquait l'erreur "not enough values to unpack").
datas = [
    ('pigoptim_app.py', '.'),
    ('pigoptim_core.py', '.'),
]
binaries = []
hiddenimports = []

for pkg in ('streamlit', 'plotly'):
    pkg_datas, pkg_binaries, pkg_hiddenimports = collect_all(pkg)
    datas += pkg_datas
    binaries += pkg_binaries
    hiddenimports += pkg_hiddenimports

a = Analysis(
    ['launcher.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    cipher=block_cipher,
)

pyz = PYZ(a.pure, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='PigOptim',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='PigOptim',
)
