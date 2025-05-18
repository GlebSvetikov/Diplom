# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_submodules, collect_data_files


hidden_imports = (
    collect_submodules('sklearn') +
    collect_submodules('imblearn') +
    collect_submodules('matplotlib')
)

datas = (
    collect_data_files('sklearn') +
    collect_data_files('imblearn') +
    collect_data_files('matplotlib')
)


icon_path = 'icon.ico'

a = Analysis(
    ['main.py'],
    pathex=['.'],  
    binaries=[],
    datas=datas,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=1,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='FlightApp',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=icon_path,
)