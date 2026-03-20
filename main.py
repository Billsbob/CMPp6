import sys
from PySide6.QtWidgets import QApplication
from UI import MainWindow

def main():
    app = QApplication(sys.argv)
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
"""
Recommended pipeline
Apply the same display/analysis transforms you want included
If the app’s clustering input is based on the rendered image state, keep those transforms consistent.
Good examples: invert, basic filters, crop/rotate if relevant.
Normalize each channel to a common scale
If you are clustering multiple visible images together, make sure each channel contributes fairly.
A simple z-score or robust scaling per channel is usually the safest choice.
Optionally add spatial coordinates
If you want clusters to be spatially coherent, include x, y features.
Scale them so they don’t overpower intensity features.
This is especially useful if you want region-like clusters instead of purely intensity-based ones.
Use PCA only if needed
If you have many channels, correlated channels, or noisy data, PCA can help.
If you only have a few channels, PCA is often unnecessary.
Then run HDBSCAN


clusterer = cuml.HDBSCAN(
    min_cluster_size=200,           # ← most important parameter
    min_samples=100,                # ← second most important
    cluster_selection_method='eom', # default & best for stable results
    cluster_selection_epsilon=0.0,  # default
    max_cluster_size=0,             # unlimited (whole layers OK)
    metric='euclidean',
    alpha=1.0,                      # default
    algorithm='best',               # cuML chooses GPU-optimal
    leaf_size=40,                   # default
    # n_jobs not needed on GPU
"""