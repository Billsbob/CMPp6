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
    # ('assets', 'assets'),  # Example: include an assets folder
]

# PyInstaller arguments
args = [
    entry_point,
    '--name=%s' % exe_name,
    '--onefile',         # Create a single executable file
    '--windowed',        # Do not open a console window (GUI app)
    '--clean',           # Clean PyInstaller cache and temporary files before building
    '--noconfirm',       # Replace output directory without asking
    '--exclude-module', 'cupy', # Exclude cupy to keep package size down
    '--exclude-module', 'cupyx',
    '--exclude-module', 'cupy_backends',
    '--exclude-module', 'cuda_pathfinder',
    '--exclude-module', 'fast_array_utils',
    '--hidden-import', 'sklearn.utils._typedefs',
    '--hidden-import', 'sklearn.neighbors._partition_nodes',
    '--hidden-import', 'sklearn.neighbors._quad_tree',
    '--hidden-import', 'sklearn.tree._utils',
    '--hidden-import', 'pynndescent',
    '--hidden-import', 'umap',
    '--hidden-import', 'igraph',
    '--hidden-import', 'leidenalg',
    '--hidden-import', 'sklearn.utils._cython_blas',
    '--hidden-import', 'sklearn.neighbors._typedefs',
    '--hidden-import', 'scipy.special.cython_special',
    '--hidden-import', 'scipy.stats._stats',
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

    # Create the spec file content if it doesn't exist or we want to ensure it's correct
    spec_content = f"""
# -*- mode: python ; coding: utf-8 -*-
import os
import sys

# Increase recursion limit for deep dependency trees like scanpy
sys.setrecursionlimit(5000)

a = Analysis(
    ['{entry_point}'],
    pathex=[],
    binaries=[],
    datas={added_data},
    hiddenimports=[
        'sklearn.utils._typedefs',
        'sklearn.neighbors._partition_nodes',
        'sklearn.neighbors._quad_tree',
        'sklearn.tree._utils',
        'sklearn.utils._cython_blas',
        'sklearn.neighbors._typedefs',
        'pynndescent',
        'umap',
        'igraph',
        'leidenalg',
        'scanpy',
        'anndata',
        'scipy.special.cython_special',
        'scipy.stats._stats'
    ],
    hookspath=[],
    hooksconfig={{}},
    runtime_hooks=[],
    excludes=['cupy', 'cupyx', 'cupy_backends', 'cuda_pathfinder', 'fast_array_utils'],
    noarchive=False,
    optimize=0,
)

# Filter out large CUDA DLLs from binaries
excluded_dlls = {{
    'cublas64_13.dll', 
    'cublasLt64_13.dll', 
    'cufft64_12.dll', 
    'curand64_10.dll', 
    'cusolver64_12.dll', 
    'cusparse64_12.dll',
    'nvJitLink_130_0.dll',
    'nvrtc64_130_0.dll'
}}

a.binaries = [x for x in a.binaries if os.path.basename(x[0]).lower() not in [d.lower() for d in excluded_dlls]]

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='{exe_name}',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
"""
    spec_filename = f"{exe_name}.spec"
    with open(spec_filename, 'w') as f:
        f.write(spec_content)

    print(f"Building {exe_name} with custom spec...")
    PyInstaller.__main__.run([spec_filename, '--noconfirm', '--clean'])
    print("Build complete.")
