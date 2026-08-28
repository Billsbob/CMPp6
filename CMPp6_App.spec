
# -*- mode: python ; coding: utf-8 -*-
import os
import sys

# Increase recursion limit for deep dependency trees
sys.setrecursionlimit(10000)

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[('README.md', '.'), ('requirements.txt', '.')],
    hiddenimports=['sklearn.utils._typedefs', 'sklearn.neighbors._partition_nodes', 'sklearn.neighbors._quad_tree', 'sklearn.tree._utils', 'sklearn.utils._cython_blas', 'scipy.special.cython_special', 'scipy.stats._stats', 'pandas._libs.tslibs.base', 'pandas._libs.tslibs.np_datetime', 'pandas._libs.tslibs.nattype', 'pandas._libs.tslibs.timedeltas', 'skimage.filters.rank.core_cy_3d', 'qimage2ndarray', 'sklearn.utils._heap', 'sklearn.utils._sorting', 'sklearn.utils._vector_sentinel'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['cupy', 'cupyx', 'cupy_backends', 'cuda_pathfinder', 'fast_array_utils'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=None,
    noarchive=False,
    module_collection_mode={
        'skimage': 'py',
        'scipy.stats': 'py',
        'sklearn': 'py',
    }
)
pyz = PYZ(a.pure, a.zipped_data, cipher=None)

# Filter out large CUDA DLLs from binaries
excluded_dlls = ('cublas64_11.dll', 'cublas64_12.dll', 'cublas64_13.dll', 'cublasLt64_11.dll', 'cublasLt64_12.dll', 'cublasLt64_13.dll', 'cufft64_10.dll', 'cufft64_11.dll', 'cufft64_12.dll', 'curand64_10.dll', 'cusolver64_11.dll', 'cusolver64_12.dll', 'cusparse64_11.dll', 'cusparse64_12.dll', 'nvJitLink_120_0.dll', 'nvJitLink_130_0.dll', 'nvrtc64_120_0.dll', 'nvrtc64_130_0.dll', 'cudnn64_8.dll', 'cudnn64_9.dll')

a.binaries = [x for x in a.binaries if os.path.basename(x[0]).lower() not in [d.lower() for d in excluded_dlls]]

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='CMPp6_App',
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
    name='CMPp6_App',
)
