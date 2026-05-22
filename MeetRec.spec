# -*- mode: python ; coding: utf-8 -*-


import os
import sys


sys.path.insert(0, os.getcwd())

from app_metadata import APP_ICON_FILENAME, ICON_IDLE_FILENAME, ICON_RECORDING_FILENAME


a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[
        (ICON_IDLE_FILENAME, 'docs'),
        (ICON_RECORDING_FILENAME, 'docs'),
    ],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    name='MeetRec',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    icon=APP_ICON_FILENAME,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    exclude_binaries=True,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='MeetRec',
)
