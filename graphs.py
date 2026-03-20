from __future__ import annotations
from typing import List, Dict, Tuple
import numpy as np
from PySide6.QtGui import QImage, QPixmap, QColor, QPainter, QPen
from PySide6.QtCore import Qt

def calculate_normalized_graphs(images: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
    """
    Calculate graphs for each image and normalize them based on the
    highest frequency among all provided images.
    
    Excludes intensity 0 from the calculation and normalization.
    
    Args:
        images: Dictionary mapping index to numpy array (grayscale image)
        
    Returns:
        Dictionary mapping index to normalized graph data (frequencies)
    """
    graphs = {}
    max_freq = 0
    
    # Calculate raw graphs
    for idx, img in images.items():
        # Calculate graph for bins 0-255
        hist, _ = np.histogram(img, bins=256, range=(0, 256))
        # Exclude intensity 0
        hist[0] = 0
        graphs[idx] = hist
        current_max = np.max(hist)
        if current_max > max_freq:
            max_freq = current_max
            
    if max_freq == 0:
        max_freq = 1
        
    # Normalize by the global maximum frequency
    normalized_graphs = {}
    for idx, hist in graphs.items():
        # Store as float for precision during drawing
        normalized_graphs[idx] = hist.astype(np.float32) / max_freq
        
    return normalized_graphs

def create_graph_pixmap(
    normalized_graph: np.ndarray, 
    color: QColor, 
    width: int, 
    height: int,
    margin: int = 50,
    draw_height_ratio: float = 0.3
) -> QPixmap:
    """
    Create a transparent QPixmap with the graph plotted.
    
    Args:
        normalized_graph: Array of 256 normalized values (0.0 to 1.0)
        color: QColor for the graph line
        width: Total width of the pixmap
        height: Total height of the pixmap
        margin: Margin around the graph
        draw_height_ratio: Ratio of the total height the graph should occupy
        
    Returns:
        QPixmap with the plotted graph
    """
    # Create a transparent QImage
    graph_overlay = QImage(width, height, QImage.Format_ARGB32)
    graph_overlay.fill(Qt.transparent)
    
    painter = QPainter(graph_overlay)
    painter.setRenderHint(QPainter.Antialiasing)
    
    draw_w = width - 2 * margin
    draw_h = int(height * draw_height_ratio)
    base_y = height - margin
    
    # Draw graph line
    pen = QPen(color)
    pen.setWidth(2)
    painter.setPen(pen)
    
    points = []
    for x in range(256):
        px = margin + (x / 255.0) * draw_w
        py = base_y - normalized_graph[x] * draw_h
        points.append((px, py))
    
    for i in range(len(points) - 1):
        painter.drawLine(
            int(points[i][0]), int(points[i][1]), 
            int(points[i+1][0]), int(points[i+1][1])
        )

    painter.end()
    return QPixmap.fromImage(graph_overlay)
