import PyInstaller.__main__
import os
import sys

# Define the entry point of your application
entry_point = 'main.py'

# Define the name of the executable
exe_name = 'CMPp6_App'

# Add any additional data files or folders needed by your application
# Format: (source, destination)
added_data = [
    ('README.md', '.'),
    ('requirements.txt', '.'),
]

# PyInstaller arguments
args = [
    entry_point,
    '--name=%s' % exe_name,
    '--onedir',          # Create a single folder with the executable and its dependencies
    '--windowed',        # Do not open a console window (GUI app)
    '--clean',           # Clean PyInstaller cache and temporary files before building
    '--noconfirm',       # Replace output directory without asking
    '--exclude-module', 'cupy', 
    '--exclude-module', 'cupyx',
    '--exclude-module', 'cupy_backends',
    '--exclude-module', 'cuda_pathfinder',
    '--exclude-module', 'fast_array_utils',
    '--hidden-import', 'sklearn.utils._typedefs',
    '--hidden-import', 'sklearn.neighbors._partition_nodes',
    '--hidden-import', 'sklearn.neighbors._quad_tree',
    '--hidden-import', 'sklearn.tree._utils',
    '--hidden-import', 'sklearn.utils._cython_blas',
    '--hidden-import', 'sklearn.neighbors._typedefs',
    '--hidden-import', 'scipy.special.cython_special',
    '--hidden-import', 'scipy.stats._stats',
    '--hidden-import', 'pandas._libs.tslibs.base',
    '--hidden-import', 'pandas._libs.tslibs.np_datetime',
    '--hidden-import', 'pandas._libs.tslibs.nattype',
    '--hidden-import', 'pandas._libs.tslibs.timedeltas',
    '--hidden-import', 'skimage.filters.rank.core_cy_3d',
    '--hidden-import', 'qimage2ndarray',
]

# Add data files to arguments
for source, dest in added_data:
    args.append('--add-data=%s%s%s' % (source, os.pathsep, dest))

# Optional: Add an icon
# icon_path = 'assets/icon.ico'
# if os.path.exists(icon_path):
#     args.append('--icon=%s' % icon_path)

# Run PyInstaller
if __name__ == '__main__':
    # Add scikit-learn and other hooks if needed
    # (PyInstaller usually handles these, but some environments need help)
    
    # Ensure dist and build folders exist
    if not os.path.exists('dist'):
        os.makedirs('dist')
    if not os.path.exists('build'):
        os.makedirs('build')

    # Hidden imports for the spec file
    hidden_imports = [
        'sklearn.utils._typedefs',
        'sklearn.neighbors._partition_nodes',
        'sklearn.neighbors._quad_tree',
        'sklearn.tree._utils',
        'sklearn.utils._cython_blas',
        'sklearn.neighbors._typedefs',
        'scipy.special.cython_special',
        'scipy.stats._stats',
        'pandas._libs.tslibs.base',
        'pandas._libs.tslibs.np_datetime',
        'pandas._libs.tslibs.nattype',
        'pandas._libs.tslibs.timedeltas',
        'skimage.filters.rank.core_cy_3d',
        'qimage2ndarray'
    ]

    # Excludes for the spec file
    excludes = ['cupy', 'cupyx', 'cupy_backends', 'cuda_pathfinder', 'fast_array_utils']

    # Create the spec file content if it doesn't exist or we want to ensure it's correct
    spec_content = f"""
# -*- mode: python ; coding: utf-8 -*-
import os
import sys

# Increase recursion limit for deep dependency trees
sys.setrecursionlimit(10000)

a = Analysis(
    ['{entry_point}'],
    pathex=[],
    binaries=[],
    datas={added_data},
    hiddenimports={hidden_imports},
    hookspath=[],
    hooksconfig={{}},
    runtime_hooks=[],
    excludes={excludes},
    noarchive=False,
    optimize=0,
)

# Filter out large CUDA DLLs from binaries
excluded_dlls = {{
    'cublas64_11.dll', 'cublas64_12.dll', 'cublas64_13.dll',
    'cublasLt64_11.dll', 'cublasLt64_12.dll', 'cublasLt64_13.dll',
    'cufft64_10.dll', 'cufft64_11.dll', 'cufft64_12.dll',
    'curand64_10.dll',
    'cusolver64_11.dll', 'cusolver64_12.dll',
    'cusparse64_11.dll', 'cusparse64_12.dll',
    'nvJitLink_120_0.dll', 'nvJitLink_130_0.dll',
    'nvrtc64_120_0.dll', 'nvrtc64_130_0.dll',
    'cudnn64_8.dll', 'cudnn64_9.dll'
}}

a.binaries = [x for x in a.binaries if os.path.basename(x[0]).lower() not in [d.lower() for d in excluded_dlls]]

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='{exe_name}',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='{exe_name}',
)
"""
    spec_filename = f"{exe_name}.spec"
    with open(spec_filename, 'w') as f:
        f.write(spec_content)

    print(f"Building {exe_name} with custom spec...")
    PyInstaller.__main__.run([spec_filename, '--noconfirm', '--clean'])
    print("Build complete.")
