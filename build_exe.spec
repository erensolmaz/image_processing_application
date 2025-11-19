# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec file for Image Processing Application

block_cipher = None

a = Analysis(
    ['app_gui_qt.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=[
        # 'PIL._tkinter_finder',  # Tkinter not required for PyQt5 build
        'cv2',
        'numpy',
        'PyQt5',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='DIP_Midterm',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # GUI uygulaması için console=False
    disable_windowed_traceback=False,
    argv_emulation=False,
    # target_arch removed - let PyInstaller detect native architecture (arm64 for Apple Silicon)
    codesign_identity=None,
    entitlements_file=None,
    icon=None,  # İsterseniz icon dosyası ekleyebilirsiniz
)

