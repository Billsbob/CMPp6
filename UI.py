from PySide6.QtWidgets import (
    QMainWindow, QMenu, QMenuBar, QFileDialog, QListWidget, QListWidgetItem, 
    QSplitter, QWidget, QVBoxLayout, QLabel, QScrollArea, QMdiArea, QMdiSubWindow, QDockWidget,
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QGraphicsRectItem, QFrame, QSlider, QInputDialog, QMessageBox,
    QPushButton, QHBoxLayout, QDialog, QFormLayout, QDoubleSpinBox, QSpinBox, QDialogButtonBox,
    QLineEdit, QComboBox, QCheckBox
)
from PySide6.QtGui import QAction, QPixmap, QIcon, QPainter, QWheelEvent, QTransform, QPalette, QPen, QColor, QBrush
from PySide6.QtCore import Qt, QSize, QPoint, QPointF, QRectF
import os
import tifffile
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from assets import AssetManager
from image_handler import ImageDisplayHandler
from clustering import apply_kmeans, apply_isodata, apply_dbscan, apply_hdbscan, apply_optics, apply_isodata_cuda, apply_dbscan_cuda, is_cuda_available

class ZoomableView(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setBackgroundRole(QPalette.NoRole)
        self.setFrameShape(QFrame.NoFrame)

        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)

        self.pixmap_item = QGraphicsPixmapItem()
        self.scene.addItem(self.pixmap_item)

        self.selection_rect_item = None
        self.start_scene_pos = None
        self.moving_selection = False
        self.move_offset = QPointF()
        self.zoom_factor = 1.0

    def set_pixmap(self, pixmap):
        if pixmap:
            self.pixmap_item.setPixmap(pixmap)
            self.scene.setSceneRect(QRectF(pixmap.rect()))
        else:
            self.pixmap_item.setPixmap(QPixmap())
            self.scene.setSceneRect(QRectF())
        self.clear_selection()

    def clear_selection(self):
        if self.selection_rect_item:
            self.scene.removeItem(self.selection_rect_item)
            self.selection_rect_item = None
        self.start_scene_pos = None

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            scene_pos = self.mapToScene(event.pos())
            
            # Check if clicking inside existing selection to move it
            if self.selection_rect_item and self.selection_rect_item.rect().contains(scene_pos):
                self.moving_selection = True
                self.move_offset = scene_pos - self.selection_rect_item.rect().topLeft()
                self.start_scene_pos = scene_pos
            else:
                self.moving_selection = False
                self.start_scene_pos = scene_pos
                if self.selection_rect_item:
                    self.scene.removeItem(self.selection_rect_item)
                    self.selection_rect_item = None
                
                self.selection_rect_item = QGraphicsRectItem()
                pen = QPen(QColor(255, 255, 0), 2)
                pen.setCosmetic(True) # Constant width regardless of zoom
                self.selection_rect_item.setPen(pen)
                self.selection_rect_item.setBrush(QBrush(QColor(255, 255, 0, 50)))
                self.scene.addItem(self.selection_rect_item)
                self.selection_rect_item.setRect(QRectF(self.start_scene_pos, QSize(0, 0)))
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.start_scene_pos is not None and self.selection_rect_item:
            current_scene_pos = self.mapToScene(event.pos())
            img_rect = self.pixmap_item.boundingRect()

            if self.moving_selection:
                # Move existing rectangle
                new_top_left = current_scene_pos - self.move_offset
                rect = self.selection_rect_item.rect()
                rect.moveTo(new_top_left)
                
                # Keep within bounds
                if rect.left() < img_rect.left():
                    rect.moveLeft(img_rect.left())
                if rect.right() > img_rect.right():
                    rect.moveRight(img_rect.right())
                if rect.top() < img_rect.top():
                    rect.moveTop(img_rect.top())
                if rect.bottom() > img_rect.bottom():
                    rect.moveBottom(img_rect.bottom())
                
                self.selection_rect_item.setRect(rect)
            else:
                # Resizing / Creating new rectangle (no square constraint)
                rect = QRectF(self.start_scene_pos, current_scene_pos).normalized()
                
                # Clip rect to image bounds
                rect = rect.intersected(img_rect)
                
                self.selection_rect_item.setRect(rect)
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.start_scene_pos = None
            self.moving_selection = False
            # Keep selection_rect_item for the crop tool
        else:
            super().mouseReleaseEvent(event)

    def get_selection_rect(self):
        if self.selection_rect_item:
            rect = self.selection_rect_item.rect()
            if rect.width() > 0 and rect.height() > 0:
                return rect
        return None

    def wheelEvent(self, event: QWheelEvent):
        if event.modifiers() & Qt.ControlModifier:
            angle = event.angleDelta().y()
            factor = 1.1 if angle > 0 else 0.9
            
            new_zoom = self.zoom_factor * factor
            if 0.1 <= new_zoom <= 20.0:
                self.zoom_factor = new_zoom
                self.scale(factor, factor)
            event.accept()
        else:
            super().wheelEvent(event)

class FilterParameterDialog(QDialog):
    def __init__(self, filter_name, initial_params, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Adjust {filter_name.capitalize()} Parameters")
        self.filter_name = filter_name
        self.params = initial_params.copy()
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        form_layout = QFormLayout()
        self.widgets = {}

        if self.filter_name == "gaussian":
            radius_spin = QDoubleSpinBox()
            radius_spin.setRange(0.1, 50.0)
            radius_spin.setValue(self.params.get("radius", 2.0))
            radius_spin.setSingleStep(0.5)
            radius_spin.valueChanged.connect(lambda v: self._update_param("radius", v))
            form_layout.addRow("Radius:", radius_spin)
            self.widgets["radius"] = radius_spin

        elif self.filter_name in ["median", "mean", "blur"]:
            size_spin = QSpinBox()
            size_spin.setRange(1, 51)
            size_spin.setSingleStep(2) # Size usually odd
            size_spin.setValue(self.params.get("size", 3))
            size_spin.valueChanged.connect(lambda v: self._update_param("size", v))
            form_layout.addRow("Size:", size_spin)
            self.widgets["size"] = size_spin

        elif self.filter_name == "unsharp":
            radius_spin = QDoubleSpinBox()
            radius_spin.setRange(0.1, 50.0)
            radius_spin.setValue(self.params.get("radius", 2.0))
            radius_spin.valueChanged.connect(lambda v: self._update_param("radius", v))
            form_layout.addRow("Radius:", radius_spin)
            self.widgets["radius"] = radius_spin

            percent_spin = QSpinBox()
            percent_spin.setRange(1, 1000)
            percent_spin.setValue(self.params.get("percent", 150))
            percent_spin.valueChanged.connect(lambda v: self._update_param("percent", v))
            form_layout.addRow("Percent:", percent_spin)
            self.widgets["percent"] = percent_spin

            threshold_spin = QSpinBox()
            threshold_spin.setRange(0, 255)
            threshold_spin.setValue(self.params.get("threshold", 3))
            threshold_spin.valueChanged.connect(lambda v: self._update_param("threshold", v))
            form_layout.addRow("Threshold:", threshold_spin)
            self.widgets["threshold"] = threshold_spin

        layout.addLayout(form_layout)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _update_param(self, key, value):
        self.params[key] = value
        # Signal real-time update if parent supports it
        if hasattr(self.parent(), "_preview_filter"):
            self.parent()._preview_filter(self.filter_name, self.params)

    def get_params(self):
        return self.params

class KMeansParameterDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("K-Means Clustering Parameters")
        self.params = {
            "n_clusters": 3,
            "max_iter": 300,
            "tol": 0.0001,
            "init": "k-means++",
            "random_state": None
        }
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        form_layout = QFormLayout()

        self.n_clusters_spin = QSpinBox()
        self.n_clusters_spin.setRange(2, 20)
        self.n_clusters_spin.setValue(self.params["n_clusters"])
        form_layout.addRow("Number of Clusters:", self.n_clusters_spin)

        self.max_iter_spin = QSpinBox()
        self.max_iter_spin.setRange(1, 10000)
        self.max_iter_spin.setValue(self.params["max_iter"])
        form_layout.addRow("Max Iterations:", self.max_iter_spin)

        self.tol_spin = QDoubleSpinBox()
        self.tol_spin.setRange(0, 1)
        self.tol_spin.setDecimals(6)
        self.tol_spin.setSingleStep(0.0001)
        self.tol_spin.setValue(self.params["tol"])
        form_layout.addRow("Tolerance:", self.tol_spin)

        self.init_combo = QComboBox()
        self.init_combo.addItems(["k-means++", "random"])
        self.init_combo.setCurrentText(self.params["init"])
        form_layout.addRow("Initial:", self.init_combo)

        self.random_state_edit = QLineEdit()
        self.random_state_edit.setPlaceholderText("None (random)")
        form_layout.addRow("Random State:", self.random_state_edit)

        layout.addLayout(form_layout)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_params(self):
        rs_text = self.random_state_edit.text()
        try:
            random_state = int(rs_text) if rs_text else None
        except ValueError:
            random_state = None
            
        return {
            "n_clusters": self.n_clusters_spin.value(),
            "max_iter": self.max_iter_spin.value(),
            "tol": self.tol_spin.value(),
            "init": self.init_combo.currentText(),
            "random_state": random_state
        }

class IsodataParameterDialog(QDialog):
    def __init__(self, parent=None, is_cuda=False):
        super().__init__(parent)
        self.is_cuda = is_cuda
        if is_cuda:
            self.setWindowTitle("CUDA ISODATA Clustering Parameters")
        else:
            self.setWindowTitle("ISODATA Clustering Parameters")
        self.params = {
            "initial_clusters": 3,
            "max_iter": 100,
            "min_samples": 20,
            "max_stddev": 10,
            "min_dist": 20,
            "max_merge_pairs": 2,
            "random_state": None
        }
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        form_layout = QFormLayout()

        self.initial_clusters_spin = QSpinBox()
        self.initial_clusters_spin.setRange(1, 100)
        self.initial_clusters_spin.setValue(self.params["initial_clusters"])
        form_layout.addRow("Initial Clusters:", self.initial_clusters_spin)

        self.max_iter_spin = QSpinBox()
        self.max_iter_spin.setRange(1, 1000)
        self.max_iter_spin.setValue(self.params["max_iter"])
        form_layout.addRow("Max Iterations:", self.max_iter_spin)

        self.min_samples_spin = QSpinBox()
        self.min_samples_spin.setRange(1, 10000)
        self.min_samples_spin.setValue(self.params["min_samples"])
        form_layout.addRow("Minimum Samples:", self.min_samples_spin)

        self.max_stddev_spin = QDoubleSpinBox()
        self.max_stddev_spin.setRange(0.1, 1000.0)
        self.max_stddev_spin.setValue(self.params["max_stddev"])
        form_layout.addRow("Max Standard Deviation:", self.max_stddev_spin)

        self.min_dist_spin = QDoubleSpinBox()
        self.min_dist_spin.setRange(0.1, 1000.0)
        self.min_dist_spin.setValue(self.params["min_dist"])
        form_layout.addRow("Minimum Cluster Distance:", self.min_dist_spin)

        self.max_merge_pairs_spin = QSpinBox()
        self.max_merge_pairs_spin.setRange(1, 50)
        self.max_merge_pairs_spin.setValue(self.params["max_merge_pairs"])
        form_layout.addRow("Max Merge Pairs:", self.max_merge_pairs_spin)

        self.random_state_edit = QLineEdit()
        self.random_state_edit.setPlaceholderText("None (random)")
        form_layout.addRow("Random State:", self.random_state_edit)

        layout.addLayout(form_layout)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_params(self):
        rs_text = self.random_state_edit.text()
        try:
            random_state = int(rs_text) if rs_text else None
        except ValueError:
            random_state = None
            
        return {
            "initial_clusters": self.initial_clusters_spin.value(),
            "max_iter": self.max_iter_spin.value(),
            "min_samples": self.min_samples_spin.value(),
            "max_stddev": self.max_stddev_spin.value(),
            "min_dist": self.min_dist_spin.value(),
            "max_merge_pairs": self.max_merge_pairs_spin.value(),
            "random_state": random_state
        }

class DBSCANParameterDialog(QDialog):
    def __init__(self, parent=None, is_cuda=False):
        super().__init__(parent)
        self.is_cuda = is_cuda
        if is_cuda:
            self.setWindowTitle("CUDA DBSCAN Clustering Parameters")
        else:
            self.setWindowTitle("DBSCAN Clustering Parameters")
        self.params = {
            "eps": 0.5,
            "min_samples": 5,
            "metric": "euclidean",
            "algorithm": "auto",
            "p": 2
        }
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        form_layout = QFormLayout()

        self.eps_spin = QDoubleSpinBox()
        self.eps_spin.setRange(0.001, 10000.0)
        self.eps_spin.setSingleStep(0.1)
        self.eps_spin.setValue(self.params["eps"])
        form_layout.addRow("Epsilon (eps):", self.eps_spin)

        self.min_samples_spin = QSpinBox()
        self.min_samples_spin.setRange(1, 10000)
        self.min_samples_spin.setValue(self.params["min_samples"])
        form_layout.addRow("Minimum Samples:", self.min_samples_spin)

        if not self.is_cuda:
            self.metric_combo = QComboBox()
            self.metric_combo.addItems(["euclidean", "manhattan", "chebyshev", "minkowski", "canberra", "braycurtis"])
            self.metric_combo.setCurrentText(self.params["metric"])
            form_layout.addRow("Metric:", self.metric_combo)

            self.algorithm_combo = QComboBox()
            self.algorithm_combo.addItems(["auto", "ball_tree", "kd_tree", "brute"])
            self.algorithm_combo.setCurrentText(self.params["algorithm"])
            form_layout.addRow("Algorithm:", self.algorithm_combo)

            self.p_spin = QDoubleSpinBox()
            self.p_spin.setRange(1.0, 100.0)
            self.p_spin.setSingleStep(1.0)
            self.p_spin.setValue(self.params["p"])
            form_layout.addRow("P (Minkowski power):", self.p_spin)

        layout.addLayout(form_layout)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_params(self):
        params = {
            "eps": self.eps_spin.value(),
            "min_samples": self.min_samples_spin.value()
        }
        if not self.is_cuda:
            params["metric"] = self.metric_combo.currentText()
            params["algorithm"] = self.algorithm_combo.currentText()
            params["p"] = self.p_spin.value()
        return params

class HDBSCANParameterDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("HDBSCAN Clustering Parameters")
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        form_layout = QFormLayout()

        self.min_cluster_size_spin = QSpinBox()
        self.min_cluster_size_spin.setRange(2, 10000)
        self.min_cluster_size_spin.setValue(5)
        form_layout.addRow("Min Cluster Size:", self.min_cluster_size_spin)

        self.min_samples_spin = QSpinBox()
        self.min_samples_spin.setRange(1, 10000)
        self.min_samples_spin.setValue(5)
        form_layout.addRow("Min Samples:", self.min_samples_spin)

        self.cluster_selection_epsilon_spin = QDoubleSpinBox()
        self.cluster_selection_epsilon_spin.setRange(0.0, 1000.0)
        self.cluster_selection_epsilon_spin.setSingleStep(0.1)
        self.cluster_selection_epsilon_spin.setValue(0.0)
        form_layout.addRow("Cluster Selection Epsilon:", self.cluster_selection_epsilon_spin)

        self.max_cluster_size_spin = QSpinBox()
        self.max_cluster_size_spin.setRange(0, 1000000)
        self.max_cluster_size_spin.setValue(0)
        form_layout.addRow("Max Cluster Size (0 for None):", self.max_cluster_size_spin)

        self.metric_combo = QComboBox()
        self.metric_combo.addItems(["euclidean", "manhattan", "chebyshev", "minkowski", "canberra", "braycurtis"])
        self.metric_combo.setCurrentText("euclidean")
        form_layout.addRow("Metric:", self.metric_combo)

        self.metric_params_edit = QLineEdit()
        form_layout.addRow("Metric Parameters (JSON):", self.metric_params_edit)

        self.alpha_spin = QDoubleSpinBox()
        self.alpha_spin.setRange(0.1, 10.0)
        self.alpha_spin.setSingleStep(0.1)
        self.alpha_spin.setValue(1.0)
        form_layout.addRow("Alpha:", self.alpha_spin)

        self.algorithm_combo = QComboBox()
        self.algorithm_combo.addItems(["auto", "brute", "kd_tree", "ball_tree"])
        self.algorithm_combo.setCurrentText("auto")
        form_layout.addRow("Algorithm:", self.algorithm_combo)

        self.leaf_size_spin = QSpinBox()
        self.leaf_size_spin.setRange(1, 1000)
        self.leaf_size_spin.setValue(40)
        form_layout.addRow("Leaf Size:", self.leaf_size_spin)

        self.n_jobs_spin = QSpinBox()
        self.n_jobs_spin.setRange(-1, 128)
        self.n_jobs_spin.setValue(-1)
        form_layout.addRow("Number of Jobs (-1 for all):", self.n_jobs_spin)

        self.cluster_selection_combo = QComboBox()
        self.cluster_selection_combo.addItems(["eom", "leaf"])
        self.cluster_selection_combo.setCurrentText("eom")
        form_layout.addRow("Cluster Selection:", self.cluster_selection_combo)

        self.allow_single_cluster_check = QCheckBox()
        self.allow_single_cluster_check.setChecked(False)
        form_layout.addRow("Allow Single Cluster:", self.allow_single_cluster_check)

        self.store_centers_combo = QComboBox()
        self.store_centers_combo.addItems(["None", "centroid", "medoid", "both"])
        self.store_centers_combo.setCurrentText("None")
        form_layout.addRow("Store Centers:", self.store_centers_combo)

        self.copy_check = QCheckBox()
        self.copy_check.setChecked(False)
        form_layout.addRow("Copy:", self.copy_check)

        layout.addLayout(form_layout)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_params(self):
        import json
        metric_params = None
        if self.metric_params_edit.text().strip():
            try:
                metric_params = json.loads(self.metric_params_edit.text())
            except:
                pass

        max_cluster_size = self.max_cluster_size_spin.value()
        if max_cluster_size == 0:
            max_cluster_size = None

        store_centers = self.store_centers_combo.currentText()
        if store_centers == "None":
            store_centers = None

        n_jobs = self.n_jobs_spin.value()
        if n_jobs == 0:
            n_jobs = None

        return {
            "min_cluster_size": self.min_cluster_size_spin.value(),
            "min_samples": self.min_samples_spin.value(),
            "cluster_selection_epsilon": self.cluster_selection_epsilon_spin.value(),
            "max_cluster_size": max_cluster_size,
            "metric": self.metric_combo.currentText(),
            "metric_params": metric_params,
            "alpha": self.alpha_spin.value(),
            "algorithm": self.algorithm_combo.currentText(),
            "leaf_size": self.leaf_size_spin.value(),
            "n_jobs": n_jobs,
            "cluster_selection_method": self.cluster_selection_combo.currentText(),
            "allow_single_cluster": self.allow_single_cluster_check.isChecked(),
            "store_centers": store_centers,
            "copy": self.copy_check.isChecked()
        }

class OPTICSParameterDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("OPTICS Clustering Parameters")
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        form_layout = QFormLayout()

        self.min_samples_spin = QSpinBox()
        self.min_samples_spin.setRange(1, 10000)
        self.min_samples_spin.setValue(5)
        form_layout.addRow("Min Samples:", self.min_samples_spin)

        self.max_eps_spin = QDoubleSpinBox()
        self.max_eps_spin.setRange(0.0, 1000000.0)
        self.max_eps_spin.setValue(1000000.0)  # Use large value for inf
        form_layout.addRow("Max Eps (Large for Inf):", self.max_eps_spin)

        self.metric_combo = QComboBox()
        self.metric_combo.addItems(["minkowski", "euclidean", "manhattan", "chebyshev", "canberra", "braycurtis"])
        self.metric_combo.setCurrentText("minkowski")
        form_layout.addRow("Metric:", self.metric_combo)

        self.p_spin = QDoubleSpinBox()
        self.p_spin.setRange(1.0, 10.0)
        self.p_spin.setValue(2.0)
        form_layout.addRow("P (for minkowski):", self.p_spin)

        self.metric_params_edit = QLineEdit()
        form_layout.addRow("Metric Parameters (JSON):", self.metric_params_edit)

        self.cluster_method_combo = QComboBox()
        self.cluster_method_combo.addItems(["xi", "dbscan"])
        self.cluster_method_combo.setCurrentText("xi")
        form_layout.addRow("Cluster Method:", self.cluster_method_combo)

        self.eps_spin = QDoubleSpinBox()
        self.eps_spin.setRange(0.0, 1000.0)
        self.eps_spin.setValue(0.0)
        form_layout.addRow("Eps (0 for None):", self.eps_spin)

        self.xi_spin = QDoubleSpinBox()
        self.xi_spin.setRange(0.0, 1.0)
        self.xi_spin.setSingleStep(0.01)
        self.xi_spin.setValue(0.05)
        form_layout.addRow("Xi:", self.xi_spin)

        self.predecessor_correction_check = QCheckBox()
        self.predecessor_correction_check.setChecked(True)
        form_layout.addRow("Predecessor Correction:", self.predecessor_correction_check)

        self.min_cluster_size_spin = QSpinBox()
        self.min_cluster_size_spin.setRange(0, 10000)
        self.min_cluster_size_spin.setValue(0)
        form_layout.addRow("Min Cluster Size (0 for None):", self.min_cluster_size_spin)

        self.algorithm_combo = QComboBox()
        self.algorithm_combo.addItems(["auto", "ball_tree", "kd_tree", "brute"])
        self.algorithm_combo.setCurrentText("auto")
        form_layout.addRow("Algorithm:", self.algorithm_combo)

        self.leaf_size_spin = QSpinBox()
        self.leaf_size_spin.setRange(1, 1000)
        self.leaf_size_spin.setValue(30)
        form_layout.addRow("Leaf Size:", self.leaf_size_spin)

        self.memory_edit = QLineEdit()
        form_layout.addRow("Memory (path or None):", self.memory_edit)

        self.n_jobs_spin = QSpinBox()
        self.n_jobs_spin.setRange(-1, 128)
        self.n_jobs_spin.setValue(-1)
        form_layout.addRow("Number of Jobs (-1 for all):", self.n_jobs_spin)

        layout.addLayout(form_layout)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_params(self):
        import json
        import numpy as np
        
        metric_params = None
        if self.metric_params_edit.text().strip():
            try:
                metric_params = json.loads(self.metric_params_edit.text())
            except:
                pass

        max_eps = self.max_eps_spin.value()
        if max_eps >= 1000000.0:
            max_eps = np.inf

        eps = self.eps_spin.value()
        if eps == 0.0:
            eps = None

        min_cluster_size = self.min_cluster_size_spin.value()
        if min_cluster_size == 0:
            min_cluster_size = None

        memory = self.memory_edit.text().strip()
        if not memory or memory.lower() == "none":
            memory = None

        n_jobs = self.n_jobs_spin.value()
        if n_jobs == 0:
            n_jobs = None

        return {
            "min_samples": self.min_samples_spin.value(),
            "max_eps": max_eps,
            "metric": self.metric_combo.currentText(),
            "p": self.p_spin.value(),
            "metric_params": metric_params,
            "cluster_method": self.cluster_method_combo.currentText(),
            "eps": eps,
            "xi": self.xi_spin.value(),
            "predecessor_correction": self.predecessor_correction_check.isChecked(),
            "min_cluster_size": min_cluster_size,
            "algorithm": self.algorithm_combo.currentText(),
            "leaf_size": self.leaf_size_spin.value(),
            "memory": memory,
            "n_jobs": n_jobs
        }

class SaveVisibleDialog(QDialog):
    def __init__(self, parent=None, has_masks=False, export_mode=None):
        super().__init__(parent)
        # export_mode can be None, 'images', or 'masks'
        if export_mode == 'images':
            self.setWindowTitle("Export Modified Images")
        elif export_mode == 'masks':
            self.setWindowTitle("Export Modified Masks")
        else:
            self.setWindowTitle("Save Visible View")
            
        self.has_masks = has_masks
        self.export_mode = export_mode
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        form = QFormLayout()

        if self.export_mode == 'images':
            default_name = "modified_images"
        elif self.export_mode == 'masks':
            default_name = "modified_masks"
        else:
            default_name = "visible_view"
            
        self.filename_edit = QLineEdit(default_name)
        form.addRow("Base Filename:" if self.export_mode else "Filename:", self.filename_edit)

        self.format_combo = QComboBox()
        if self.export_mode == 'masks':
            self.format_combo.addItems(["bmp", "png", "tif", "jpg"])
        else:
            self.format_combo.addItems(["tif", "png", "jpg", "bmp"])
        form.addRow("Image Format:", self.format_combo)

        layout.addLayout(form)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_options(self):
        options = {
            "filename": self.filename_edit.text(),
            "format": self.format_combo.currentText()
        }
        return options

class HistogramDialog(QDialog):
    def __init__(self, images, masks, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Generate Histogram")
        self.images = images
        self.masks = masks
        self.setup_ui()

    def setup_ui(self):
        layout = QFormLayout(self)

        self.image_combo = QComboBox()
        self.image_combo.addItems(self.images)
        layout.addRow("Select Image:", self.image_combo)

        self.mask_combo = QComboBox()
        self.mask_combo.addItems(self.masks)
        layout.addRow("Select Mask:", self.mask_combo)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

    def get_selection(self):
        return self.image_combo.currentText(), self.mask_combo.currentText()

class HistogramViewer(QWidget):
    def __init__(self, data, title="Histogram", parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.figure)
        self.layout.addWidget(self.canvas)
        
        self.ax.hist(data.flatten(), bins=256, color='gray', alpha=0.7)
        self.ax.set_title(title)
        self.ax.set_xlabel("Pixel Value")
        self.ax.set_ylabel("Frequency")
        self.canvas.draw()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Clustering App")
        self.resize(1500, 1000)

        self.asset_manager = AssetManager()
        self.image_handler = ImageDisplayHandler()
        self.working_dir = None
        
        self.cached_composite = None

        self._create_menu_bar()
        self._create_status_bar()
        self._setup_ui()

    def _setup_ui(self):
        # Create the MDI workspace
        self.mdi_area = QMdiArea()
        self.setCentralWidget(self.mdi_area)

        # Create a dock widget for images and masks
        dock = QDockWidget("Assets", self)
        dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.addDockWidget(Qt.LeftDockWidgetArea, dock)

        # Left side in a vertical layout for the dock
        dock_container = QWidget()
        dock_layout = QVBoxLayout(dock_container)
        
        # Image list
        image_container = QWidget()
        image_h_layout = QHBoxLayout(image_container)
        image_h_layout.setContentsMargins(0, 0, 0, 0)
        
        image_btn_layout = QVBoxLayout()
        image_btn_layout.setContentsMargins(0, 0, 5, 0)
        
        self.select_all_images_btn = QPushButton("Select\nAll")
        self.select_all_images_btn.setFixedWidth(60)
        self.select_all_images_btn.clicked.connect(self._select_all_images)
        self.select_none_images_btn = QPushButton("Select\nNone")
        self.select_none_images_btn.setFixedWidth(60)
        self.select_none_images_btn.clicked.connect(self._select_none_images)
        
        image_btn_layout.addWidget(self.select_all_images_btn)
        image_btn_layout.addWidget(self.select_none_images_btn)
        image_btn_layout.addStretch()
        
        self.image_list = QListWidget()
        self.image_list.setIconSize(QSize(100, 100))
        self.image_list.setSelectionMode(QListWidget.MultiSelection)
        self.image_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.image_list.customContextMenuRequested.connect(self._show_image_context_menu)
        self.image_list.itemClicked.connect(self._asset_clicked)
        
        image_h_layout.addLayout(image_btn_layout)
        image_h_layout.addWidget(self.image_list)
        
        dock_layout.addWidget(QLabel("Images:"))
        dock_layout.addWidget(image_container)

        # Mask list
        mask_container = QWidget()
        mask_h_layout = QHBoxLayout(mask_container)
        mask_h_layout.setContentsMargins(0, 0, 0, 0)
        
        mask_btn_layout = QVBoxLayout()
        mask_btn_layout.setContentsMargins(0, 0, 5, 0)
        
        self.select_all_masks_btn = QPushButton("Select\nAll")
        self.select_all_masks_btn.setFixedWidth(60)
        self.select_all_masks_btn.clicked.connect(self._select_all_masks)
        self.select_none_masks_btn = QPushButton("Select\nNone")
        self.select_none_masks_btn.setFixedWidth(60)
        self.select_none_masks_btn.clicked.connect(self._select_none_masks)
        
        mask_btn_layout.addWidget(self.select_all_masks_btn)
        mask_btn_layout.addWidget(self.select_none_masks_btn)
        mask_btn_layout.addStretch()
        
        self.mask_list = QListWidget()
        self.mask_list.setIconSize(QSize(100, 100))
        self.mask_list.setSelectionMode(QListWidget.MultiSelection)
        self.mask_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.mask_list.customContextMenuRequested.connect(self._show_mask_context_menu)
        self.mask_list.itemClicked.connect(self._mask_clicked)
        
        mask_h_layout.addLayout(mask_btn_layout)
        mask_h_layout.addWidget(self.mask_list)
        
        dock_layout.addWidget(QLabel("Masks:"))
        dock_layout.addWidget(mask_container)

        # Histogram list
        self.hist_list = QListWidget()
        self.hist_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.hist_list.customContextMenuRequested.connect(self._show_hist_context_menu)
        self.hist_list.itemDoubleClicked.connect(self._hist_item_double_clicked)
        
        dock_layout.addWidget(QLabel("Histograms:"))
        dock_layout.addWidget(self.hist_list)

        # Mask opacity slider
        dock_layout.addWidget(QLabel("Mask Opacity:"))
        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setMinimum(0)
        self.opacity_slider.setMaximum(100)
        self.opacity_slider.setValue(50)
        self.opacity_slider.valueChanged.connect(self._opacity_changed)
        dock_layout.addWidget(self.opacity_slider)

        # Save button
        self.save_button = QPushButton("Save Visible")
        self.save_button.clicked.connect(self._save_visible)
        dock_layout.addWidget(self.save_button)

        dock.setWidget(dock_container)

        # Create the image viewer in a sub-window
        self.viewer_subwindow = QMdiSubWindow()
        self.viewer_subwindow.setWindowTitle("Image Viewer")
        
        self.viewer_view = ZoomableView()
        
        self.viewer_subwindow.setWidget(self.viewer_view)
        self.mdi_area.addSubWindow(self.viewer_subwindow)
        self.viewer_subwindow.show()

    def _create_menu_bar(self):
        menu_bar = self.menuBar()

        # "Home" button - as an action directly in the menu bar if possible, 
        # but typically menus contain dropdowns. 
        # However, many apps use a single action for Home.
        home_action = QAction("Home", self)
        home_action.triggered.connect(self._home_triggered)
        menu_bar.addAction(home_action)

        # "Cluster" dropdown button
        cluster_menu = menu_bar.addMenu("Cluster")
        kmeans_action = QAction("K-Means", self)
        kmeans_action.triggered.connect(self._apply_kmeans_triggered)
        cluster_menu.addAction(kmeans_action)

        isodata_action = QAction("ISODATA", self)
        isodata_action.triggered.connect(self._apply_isodata_triggered)
        cluster_menu.addAction(isodata_action)

        cuda_isodata_action = QAction("CUDA ISODATA", self)
        cuda_isodata_action.triggered.connect(self._apply_cuda_isodata_triggered)
        if not is_cuda_available():
            cuda_isodata_action.setEnabled(False)
            cuda_isodata_action.setToolTip("CUDA not available")
        cluster_menu.addAction(cuda_isodata_action)

        dbscan_action = QAction("DBSCAN", self)
        dbscan_action.triggered.connect(self._apply_dbscan_triggered)
        cluster_menu.addAction(dbscan_action)

        cuda_dbscan_action = QAction("CUDA DBSCAN", self)
        cuda_dbscan_action.triggered.connect(self._apply_cuda_dbscan_triggered)
        if not is_cuda_available():
            cuda_dbscan_action.setEnabled(False)
            cuda_dbscan_action.setToolTip("CUDA not available")
        cluster_menu.addAction(cuda_dbscan_action)

        hdbscan_action = QAction("HDBSCAN", self)
        hdbscan_action.triggered.connect(self._apply_hdbscan_triggered)
        cluster_menu.addAction(hdbscan_action)

        optics_action = QAction("OPTICS", self)
        optics_action.triggered.connect(self._apply_optics_triggered)
        cluster_menu.addAction(optics_action)

        # "Tools" dropdown button
        tools_menu = menu_bar.addMenu("Tools")
        invert_action = QAction("Invert", self)
        invert_action.triggered.connect(self._invert_selected_images)
        tools_menu.addAction(invert_action)

        crop_action = QAction("Crop All", self)
        crop_action.triggered.connect(self._crop_all)
        tools_menu.addAction(crop_action)

        rotate_action = QAction("Rotate All", self)
        rotate_action.triggered.connect(self._rotate_all)
        tools_menu.addAction(rotate_action)

        merge_masks_action = QAction("Merge Masks", self)
        merge_masks_action.triggered.connect(self._merge_masks_triggered)
        tools_menu.addAction(merge_masks_action)

        histogram_action = QAction("Generate Histogram", self)
        histogram_action.triggered.connect(self._generate_histogram_triggered)
        tools_menu.addAction(histogram_action)

        filters_menu = tools_menu.addMenu("Filters")
        
        gaussian_action = QAction("Gaussian", self)
        gaussian_action.triggered.connect(lambda: self._apply_filter_to_visible("gaussian"))
        filters_menu.addAction(gaussian_action)
        
        median_action = QAction("Median", self)
        median_action.triggered.connect(lambda: self._apply_filter_to_visible("median"))
        filters_menu.addAction(median_action)
        
        mean_action = QAction("Mean", self)
        mean_action.triggered.connect(lambda: self._apply_filter_to_visible("mean"))
        filters_menu.addAction(mean_action)
        
        blur_action = QAction("Blur", self)
        blur_action.triggered.connect(lambda: self._apply_filter_to_visible("blur"))
        filters_menu.addAction(blur_action)
        
        unsharp_action = QAction("Unsharp", self)
        unsharp_action.triggered.connect(lambda: self._apply_filter_to_visible("unsharp"))
        filters_menu.addAction(unsharp_action)

        tools_menu.addSeparator()
        export_images_action = QAction("Export Modified Images", self)
        export_images_action.triggered.connect(self._export_modified_images)
        tools_menu.addAction(export_images_action)
        
        export_masks_action = QAction("Export Modified Masks", self)
        export_masks_action.triggered.connect(self._export_modified_masks)
        tools_menu.addAction(export_masks_action)

    def _home_triggered(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Working Directory")
        if directory:
            self.working_dir = directory
            self.asset_manager.set_working_dir(directory)
            self._update_asset_list()
            self.statusBar().showMessage(f"Working Directory: {self.working_dir}")
            print(f"Working directory set to: {self.working_dir}")

    def _update_asset_list(self):
        self.image_list.clear()
        for name in self.asset_manager.get_image_list():
            image_asset, _ = self.asset_manager.get_asset_pair(name)
            
            # Create thumbnail
            qimg = image_asset.to_qimage()
            pixmap = QPixmap.fromImage(qimg)
            thumbnail = pixmap.scaled(100, 100, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            
            item = QListWidgetItem(name)
            item.setIcon(QIcon(thumbnail))
            
            # Use color in item display (optional but good for UX)
            color_name = self.image_handler.get_asset_color(name)
            item.setToolTip(f"Color: {color_name}")
            
            self.image_list.addItem(item)
            
            # Restore visibility status if any
            if self.image_handler.is_visible(name):
                item.setSelected(True)

        self.mask_list.clear()
        for name in self.asset_manager.get_mask_list():
            mask_asset = None
            for m in self.asset_manager.masks.values():
                if m.name == name:
                    mask_asset = m
                    break
            
            if not mask_asset:
                continue
            
            # Create thumbnail
            qimg = mask_asset.to_qimage()
            pixmap = QPixmap.fromImage(qimg)
            thumbnail = pixmap.scaled(100, 100, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            
            item = QListWidgetItem(name)
            item.setIcon(QIcon(thumbnail))
            self.mask_list.addItem(item)
            
            # Restore visibility status
            if self.image_handler.is_visible(name, is_mask=True):
                item.setSelected(True)

        # Update histogram list
        self.hist_list.clear()
        if self.working_dir:
            hist_dir = os.path.join(self.working_dir, "histograms")
            if os.path.exists(hist_dir):
                for filename in os.listdir(hist_dir):
                    if filename.endswith(".png"):
                        item = QListWidgetItem(filename)
                        # Optionally add thumbnail for histogram
                        self.hist_list.addItem(item)

    def _show_image_context_menu(self, position: QPoint):
        self._show_context_menu(self.image_list, position, is_mask=False)

    def _show_mask_context_menu(self, position: QPoint):
        self._show_context_menu(self.mask_list, position, is_mask=True)

    def _show_context_menu(self, list_widget, position, is_mask):
        item = list_widget.itemAt(position)
        if not item:
            return

        name = item.text()
        menu = QMenu()
        
        # Rename
        rename_action = QAction("Rename", self)
        rename_action.triggered.connect(lambda: self._rename_asset(name, is_mask))
        menu.addAction(rename_action)
        menu.addSeparator()

        color_menu = menu.addMenu("Change Color")
        for color_name in self.image_handler.COLORS.keys():
            action = QAction(color_name.capitalize(), self)
            action.triggered.connect(lambda checked=False, n=name, c=color_name, m=is_mask: self._change_color(n, c, m))
            color_menu.addAction(action)

        # Transforms
        if not is_mask:
            image_asset, _ = self.asset_manager.get_asset_pair(name)
            if image_asset:
                cfg = image_asset.pipeline.config
                menu.addSeparator()
                
                stretch_action = QAction("Contrast Stretch", self)
                stretch_action.setCheckable(True)
                stretch_action.setChecked(cfg.get("contrast_stretch", False))
                stretch_action.triggered.connect(lambda: self._toggle_transform(image_asset, "contrast_stretch"))
                menu.addAction(stretch_action)
            
        menu.exec(list_widget.mapToGlobal(position))

    def _show_hist_context_menu(self, position):
        item = self.hist_list.itemAt(position)
        if not item:
            return

        filename = item.text()
        menu = QMenu()
        
        open_action = QAction("Open", self)
        open_action.triggered.connect(lambda: self._open_hist_viewer(filename))
        menu.addAction(open_action)

        rename_action = QAction("Rename", self)
        rename_action.triggered.connect(lambda: self._rename_hist(filename))
        menu.addAction(rename_action)

        save_csv_action = QAction("Save .csv", self)
        save_csv_action.triggered.connect(lambda: self._save_hist_csv(filename))
        menu.addAction(save_csv_action)

        delete_action = QAction("Delete", self)
        delete_action.triggered.connect(lambda: self._delete_hist(filename))
        menu.addAction(delete_action)

        menu.exec(self.hist_list.mapToGlobal(position))

    def _hist_item_double_clicked(self, item):
        self._open_hist_viewer(item.text())

    def _open_hist_viewer(self, filename):
        path = os.path.join(self.working_dir, "histograms", filename)
        if os.path.exists(path):
            from PySide6.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsPixmapItem
            pixmap = QPixmap(path)
            
            view = ZoomableView()
            view.set_pixmap(pixmap)
            
            subwindow = QMdiSubWindow()
            subwindow.setWidget(view)
            subwindow.setWindowTitle(f"Histogram View - {filename}")
            self.mdi_area.addSubWindow(subwindow)
            subwindow.show()

    def _rename_hist(self, filename):
        new_name, ok = QInputDialog.getText(self, "Rename Histogram", "New name:", QLineEdit.Normal, filename)
        if ok and new_name:
            if not new_name.endswith(".png"):
                new_name += ".png"
            
            old_path = os.path.join(self.working_dir, "histograms", filename)
            new_path = os.path.join(self.working_dir, "histograms", new_name)
            
            try:
                os.rename(old_path, new_path)
                # Also rename CSV if exists
                old_csv = old_path.replace(".png", ".csv")
                new_csv = new_path.replace(".png", ".csv")
                if os.path.exists(old_csv):
                    os.rename(old_csv, new_csv)
                self._update_asset_list()
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Could not rename: {str(e)}")

    def _save_hist_csv(self, filename):
        csv_path = os.path.join(self.working_dir, "histograms", filename.replace(".png", ".csv"))
        if os.path.exists(csv_path):
            save_path, _ = QFileDialog.getSaveFileName(self, "Save CSV", filename.replace(".png", ".csv"), "CSV Files (*.csv)")
            if save_path:
                import shutil
                shutil.copy(csv_path, save_path)
        else:
            QMessageBox.warning(self, "Warning", "CSV data not found for this histogram.")

    def _delete_hist(self, filename):
        reply = QMessageBox.question(self, "Delete", f"Are you sure you want to delete {filename}?", QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            path = os.path.join(self.working_dir, "histograms", filename)
            csv_path = path.replace(".png", ".csv")
            try:
                os.remove(path)
                if os.path.exists(csv_path):
                    os.remove(csv_path)
                self._update_asset_list()
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Could not delete: {str(e)}")

    def _generate_histogram_triggered(self):
        images = self.asset_manager.get_image_list()
        masks = self.asset_manager.get_mask_list()
        
        if not images or not masks:
            QMessageBox.warning(self, "Warning", "Need at least one image and one mask.")
            return

        dialog = HistogramDialog(images, masks, self)
        if dialog.exec() == QDialog.Accepted:
            image_name, mask_name = dialog.get_selection()
            
            image_asset, _ = self.asset_manager.get_asset_pair(image_name)
            mask_asset = None
            for m in self.asset_manager.masks.values():
                if m.name == mask_name:
                    mask_asset = m
                    break
            
            if image_asset and mask_asset:
                img_data = image_asset.data
                mask_data = mask_asset.data
                
                # Apply mask (ROI)
                roi_pixels = img_data[mask_data > 0]
                
                if roi_pixels.size == 0:
                    QMessageBox.warning(self, "Warning", "The mask does not cover any pixels.")
                    return

                # Create viewer
                viewer = HistogramViewer(roi_pixels, title=f"Histogram: {image_name} (Mask: {mask_name})")
                subwindow = QMdiSubWindow()
                subwindow.setWidget(viewer)
                subwindow.setWindowTitle(f"Histogram - {image_name}")
                self.mdi_area.addSubWindow(subwindow)
                subwindow.show()

                # Save copy to "histograms" folder
                if self.working_dir:
                    hist_dir = os.path.join(self.working_dir, "histograms")
                    os.makedirs(hist_dir, exist_ok=True)
                    
                    base_name = f"{image_name}_{mask_name}_hist"
                    img_save_path = os.path.join(hist_dir, f"{base_name}.png")
                    # To avoid overwrite, maybe add timestamp or counter
                    counter = 1
                    while os.path.exists(img_save_path):
                        img_save_path = os.path.join(hist_dir, f"{base_name}_{counter}.png")
                        counter += 1
                    
                    viewer.figure.savefig(img_save_path)
                    
                    # Save .csv
                    csv_save_path = img_save_path.replace(".png", ".csv")
                    # Calculate histogram values for CSV
                    hist_vals, bin_edges = np.histogram(roi_pixels.flatten(), bins=256)
                    df = pd.DataFrame({'bin_start': bin_edges[:-1], 'count': hist_vals})
                    df.to_csv(csv_save_path, index=False)
                    
                    self._update_asset_list()

    def _invert_selected_images(self):
        """Inverts all images that are currently selected (visible)."""
        visible_images = list(self.image_handler.visible_assets)
        if not visible_images:
            return

        for name in visible_images:
            image_asset, _ = self.asset_manager.get_asset_pair(name)
            if image_asset:
                # Toggle invert in pipeline config
                current_invert = image_asset.pipeline.config.get("invert", False)
                image_asset.pipeline.config["invert"] = not current_invert
                image_asset.save_project()
        
        # Clear cache and refresh viewer to show changes
        self.cached_composite = None
        self._refresh_viewer()

    def _toggle_transform(self, asset, key):
        asset.pipeline.config[key] = not asset.pipeline.config.get(key, False)
        asset.save_project()
        self.cached_composite = None
        self._refresh_viewer()

    def _apply_filter_to_visible(self, filter_name):
        # Apply filter to images toggled visible
        visible_assets = []
        for name in self.image_handler.visible_assets:
            image_asset, _ = self.asset_manager.get_asset_pair(name)
            if image_asset:
                visible_assets.append(image_asset)
        
        if not visible_assets:
            QMessageBox.information(self, "No Images Visible", "Please toggle at least one image visible to apply a filter.")
            return

        # Get initial params from the first visible asset (if any)
        initial_params = visible_assets[0].pipeline.config.get("filter_params", {}).get(filter_name, {})
        if not initial_params:
            # Provide defaults if not set
            if filter_name == "gaussian": initial_params = {"radius": 2.0}
            elif filter_name in ["median", "mean", "blur"]: initial_params = {"size": 3}
            elif filter_name == "unsharp": initial_params = {"radius": 2.0, "percent": 150, "threshold": 3}

        # Open dialog for parameters
        dialog = FilterParameterDialog(filter_name, initial_params, self)
        
        # Store original configs for cancelation
        original_configs = [(asset, asset.pipeline.config.copy()) for asset in visible_assets]

        if dialog.exec() == QDialog.Accepted:
            final_params = dialog.get_params()
            for asset in visible_assets:
                filters = asset.pipeline.config.get("filters", [])
                if filter_name not in filters:
                    filters.append(filter_name)
                asset.pipeline.config["filters"] = filters
                
                filter_params = asset.pipeline.config.get("filter_params", {})
                filter_params[filter_name] = final_params
                asset.pipeline.config["filter_params"] = filter_params
                
                asset.save_project()
            
            self._update_asset_list() # To refresh thumbnails
            self.cached_composite = None
            self._refresh_viewer()
        else:
            # Restore original configs
            for asset, config in original_configs:
                asset.pipeline.config = config
            self.cached_composite = None
            self._refresh_viewer()

    def _apply_kmeans_triggered(self):
        visible_names = list(self.image_handler.visible_assets)
        if not visible_names:
            QMessageBox.information(self, "No Images Visible", "Please toggle at least one image visible for clustering.")
            return

        dialog = KMeansParameterDialog(self)
        if dialog.exec() != QDialog.Accepted:
            return

        params = dialog.get_params()
        
        # Collect data from visible images
        stack = []
        for name in sorted(visible_names):
            image_asset, _ = self.asset_manager.get_asset_pair(name)
            if image_asset:
                data = image_asset.get_rendered_data(for_clustering=True)
                if data is not None:
                    stack.append(data)
        
        if not stack:
            return
            
        # Combine stacked data if multiple images
        if len(stack) > 1:
            # Ensure all have same shape
            # In a real app we might need to resize, here assume same size
            combined_data = np.stack([s.astype(np.float32) for s in stack], axis=-1)
        else:
            combined_data = stack[0]

        try:
            labels = apply_kmeans(combined_data, **params)
            
            # Create masks for each cluster
            home_folder_name = os.path.basename(self.working_dir) if self.working_dir else "Project"
            
            for i in range(params["n_clusters"]):
                cluster_mask = (labels == i).astype(np.uint8) * 255
                mask_name = f"{home_folder_name}_KC_{i:02d}"
                
                # Create mask asset and add to manager
                self.asset_manager.add_new_mask(mask_name, cluster_mask)
                
            self._update_asset_list()
            self.statusBar().showMessage(f"K-Means clustering complete. Created {params['n_clusters']} masks.")
            
        except Exception as e:
            QMessageBox.critical(self, "K-Means Error", f"An error occurred during clustering: {str(e)}")

    def _apply_isodata_triggered(self):
        visible_names = list(self.image_handler.visible_assets)
        if not visible_names:
            QMessageBox.information(self, "No Images Visible", "Please toggle at least one image visible for clustering.")
            return

        dialog = IsodataParameterDialog(self)
        if dialog.exec() != QDialog.Accepted:
            return

        params = dialog.get_params()
        
        # Collect data from visible images
        stack = []
        for name in sorted(visible_names):
            image_asset, _ = self.asset_manager.get_asset_pair(name)
            if image_asset:
                data = image_asset.get_rendered_data(for_clustering=True)
                if data is not None:
                    stack.append(data)
        
        if not stack:
            return
            
        # Combine stacked data if multiple images
        if len(stack) > 1:
            combined_data = np.stack([s.astype(np.float32) for s in stack], axis=-1)
        else:
            combined_data = stack[0]

        try:
            labels = apply_isodata(combined_data, **params)
            
            # Create masks for each cluster
            home_folder_name = os.path.basename(self.working_dir) if self.working_dir else "Project"
            unique_labels = np.unique(labels)
            
            for i, label in enumerate(unique_labels):
                cluster_mask = (labels == label).astype(np.uint8) * 255
                mask_name = f"{home_folder_name}_IC_{i:02d}"
                
                # Create mask asset and add to manager
                self.asset_manager.add_new_mask(mask_name, cluster_mask)
                
            self._update_asset_list()
            self.statusBar().showMessage(f"ISODATA clustering complete. Created {len(unique_labels)} masks.")
            
        except Exception as e:
            QMessageBox.critical(self, "ISODATA Error", f"An error occurred during clustering: {str(e)}")

    def _apply_cuda_isodata_triggered(self):
        visible_names = list(self.image_handler.visible_assets)
        if not visible_names:
            QMessageBox.information(self, "No Images Visible", "Please toggle at least one image visible for clustering.")
            return

        dialog = IsodataParameterDialog(self, is_cuda=True)
        if dialog.exec() != QDialog.Accepted:
            return

        params = dialog.get_params()
        
        # Collect data from visible images
        stack = []
        for name in sorted(visible_names):
            image_asset, _ = self.asset_manager.get_asset_pair(name)
            if image_asset:
                data = image_asset.get_rendered_data(for_clustering=True)
                if data is not None:
                    stack.append(data)
        
        if not stack:
            return
            
        # Combine stacked data if multiple images
        if len(stack) > 1:
            combined_data = np.stack([s.astype(np.float32) for s in stack], axis=-1)
        else:
            combined_data = stack[0]

        try:
            labels = apply_isodata_cuda(combined_data, **params)
            
            # Create masks for each cluster
            home_folder_name = os.path.basename(self.working_dir) if self.working_dir else "Project"
            unique_labels = np.unique(labels)
            
            for i, label in enumerate(unique_labels):
                cluster_mask = (labels == label).astype(np.uint8) * 255
                mask_name = f"{home_folder_name}_CUDA_IC_{i:02d}"
                
                # Create mask asset and add to manager
                self.asset_manager.add_new_mask(mask_name, cluster_mask)
                
            self._update_asset_list()
            self.statusBar().showMessage(f"CUDA ISODATA clustering complete. Created {len(unique_labels)} masks.")
            
        except Exception as e:
            QMessageBox.critical(self, "CUDA ISODATA Error", f"An error occurred during clustering: {str(e)}")

    def _apply_dbscan_triggered(self):
        visible_names = list(self.image_handler.visible_assets)
        if not visible_names:
            QMessageBox.information(self, "No Images Visible", "Please toggle at least one image visible for clustering.")
            return

        dialog = DBSCANParameterDialog(self)
        if dialog.exec() != QDialog.Accepted:
            return

        params = dialog.get_params()
        
        # Collect data from visible images
        stack = []
        for name in sorted(visible_names):
            image_asset, _ = self.asset_manager.get_asset_pair(name)
            if image_asset:
                data = image_asset.get_rendered_data(for_clustering=True)
                if data is not None:
                    stack.append(data)
        
        if not stack:
            return
            
        # Combine stacked data if multiple images
        if len(stack) > 1:
            combined_data = np.stack([s.astype(np.float32) for s in stack], axis=-1)
        else:
            combined_data = stack[0]

        try:
            labels = apply_dbscan(combined_data, **params)
            
            # Create masks for each cluster
            home_folder_name = os.path.basename(self.working_dir) if self.working_dir else "Project"
            unique_labels = np.unique(labels)
            
            # unique_labels might include -1 for noise
            for i, label in enumerate(unique_labels):
                cluster_mask = (labels == label).astype(np.uint8) * 255
                if label == -1:
                    mask_name = f"{home_folder_name}_DC_Noise"
                else:
                    mask_name = f"{home_folder_name}_DC_{i:02d}"
                
                # Create mask asset and add to manager
                self.asset_manager.add_new_mask(mask_name, cluster_mask)
                
            self._update_asset_list()
            self.statusBar().showMessage(f"DBSCAN clustering complete. Created {len(unique_labels)} masks.")
            
        except Exception as e:
            QMessageBox.critical(self, "DBSCAN Error", f"An error occurred during clustering: {str(e)}")

    def _apply_cuda_dbscan_triggered(self):
        visible_names = list(self.image_handler.visible_assets)
        if not visible_names:
            QMessageBox.information(self, "No Images Visible", "Please toggle at least one image visible for clustering.")
            return

        dialog = DBSCANParameterDialog(self, is_cuda=True)
        if dialog.exec() != QDialog.Accepted:
            return

        params = dialog.get_params()
        
        # Collect data from visible images
        stack = []
        for name in sorted(visible_names):
            image_asset, _ = self.asset_manager.get_asset_pair(name)
            if image_asset:
                data = image_asset.get_rendered_data(for_clustering=True)
                if data is not None:
                    stack.append(data)
        
        if not stack:
            return
            
        # Combine stacked data if multiple images
        if len(stack) > 1:
            combined_data = np.stack([s.astype(np.float32) for s in stack], axis=-1)
        else:
            combined_data = stack[0]

        try:
            labels = apply_dbscan_cuda(combined_data, **params)
            
            # Create masks for each cluster
            home_folder_name = os.path.basename(self.working_dir) if self.working_dir else "Project"
            unique_labels = np.unique(labels)
            
            for i, label in enumerate(unique_labels):
                cluster_mask = (labels == label).astype(np.uint8) * 255
                if label == -1:
                    mask_name = f"{home_folder_name}_CUDA_DC_Noise"
                else:
                    mask_name = f"{home_folder_name}_CUDA_DC_{i:02d}"
                
                # Create mask asset and add to manager
                self.asset_manager.add_new_mask(mask_name, cluster_mask)
                
            self._update_asset_list()
            self.statusBar().showMessage(f"CUDA DBSCAN clustering complete. Created {len(unique_labels)} masks.")
            
        except Exception as e:
            QMessageBox.critical(self, "CUDA DBSCAN Error", f"An error occurred during clustering: {str(e)}")

    def _apply_hdbscan_triggered(self):
        visible_names = list(self.image_handler.visible_assets)
        if not visible_names:
            QMessageBox.information(self, "No Images Visible", "Please toggle at least one image visible for clustering.")
            return

        dialog = HDBSCANParameterDialog(self)
        if dialog.exec() != QDialog.Accepted:
            return

        params = dialog.get_params()
        
        # Collect data from visible images
        stack = []
        for name in sorted(visible_names):
            image_asset, _ = self.asset_manager.get_asset_pair(name)
            if image_asset:
                data = image_asset.get_rendered_data(for_clustering=True)
                if data is not None:
                    stack.append(data)
        
        if not stack:
            return
            
        # Combine stacked data if multiple images
        if len(stack) > 1:
            combined_data = np.stack([s.astype(np.float32) for s in stack], axis=-1)
        else:
            combined_data = stack[0]

        try:
            labels = apply_hdbscan(combined_data, **params)
            
            # Create masks for each cluster
            home_folder_name = os.path.basename(self.working_dir) if self.working_dir else "Project"
            unique_labels = np.unique(labels)
            
            for i, label in enumerate(unique_labels):
                cluster_mask = (labels == label).astype(np.uint8) * 255
                if label == -1:
                    mask_name = f"{home_folder_name}_HC_Noise"
                else:
                    mask_name = f"{home_folder_name}_HC_{i:02d}"
                
                # Create mask asset and add to manager
                self.asset_manager.add_new_mask(mask_name, cluster_mask)
                
            self._update_asset_list()
            self.statusBar().showMessage(f"HDBSCAN clustering complete. Created {len(unique_labels)} masks.")
            
        except Exception as e:
            QMessageBox.critical(self, "HDBSCAN Error", f"An error occurred during clustering: {str(e)}")

    def _apply_optics_triggered(self):
        visible_names = list(self.image_handler.visible_assets)
        if not visible_names:
            QMessageBox.information(self, "No Images Visible", "Please toggle at least one image visible for clustering.")
            return

        dialog = OPTICSParameterDialog(self)
        if dialog.exec() != QDialog.Accepted:
            return

        params = dialog.get_params()
        
        # Collect data from visible images
        stack = []
        for name in sorted(visible_names):
            image_asset, _ = self.asset_manager.get_asset_pair(name)
            if image_asset:
                data = image_asset.get_rendered_data(for_clustering=True)
                if data is not None:
                    stack.append(data)
        
        if not stack:
            return
            
        # Combine stacked data if multiple images
        if len(stack) > 1:
            combined_data = np.stack([s.astype(np.float32) for s in stack], axis=-1)
        else:
            combined_data = stack[0]

        try:
            labels = apply_optics(combined_data, **params)
            
            # Create masks for each cluster
            home_folder_name = os.path.basename(self.working_dir) if self.working_dir else "Project"
            unique_labels = np.unique(labels)
            
            for i, label in enumerate(unique_labels):
                cluster_mask = (labels == label).astype(np.uint8) * 255
                if label == -1:
                    mask_name = f"{home_folder_name}_OC_Noise"
                else:
                    mask_name = f"{home_folder_name}_OC_{i:02d}"
                
                # Create mask asset and add to manager
                self.asset_manager.add_new_mask(mask_name, cluster_mask)
                
            self._update_asset_list()
            self.statusBar().showMessage(f"OPTICS clustering complete. Created {len(unique_labels)} masks.")
            
        except Exception as e:
            QMessageBox.critical(self, "OPTICS Error", f"An error occurred during clustering: {str(e)}")

    def _preview_filter(self, filter_name, params):
        # Temporary application for real-time preview
        for name in self.image_handler.visible_assets:
            image_asset, _ = self.asset_manager.get_asset_pair(name)
            if image_asset:
                filters = image_asset.pipeline.config.get("filters", [])
                if filter_name not in filters:
                    filters.append(filter_name)
                image_asset.pipeline.config["filters"] = filters
                
                filter_params = image_asset.pipeline.config.get("filter_params", {})
                filter_params[filter_name] = params
                image_asset.pipeline.config["filter_params"] = filter_params
                
        self.cached_composite = None
        self._refresh_viewer()


    def _rename_asset(self, old_name, is_mask):
        new_name, ok = QInputDialog.getText(self, "Rename Asset", "Enter new name:", text=old_name)
        if ok and new_name and new_name != old_name:
            if is_mask:
                success, result = self.asset_manager.rename_mask(old_name, new_name)
                if success:
                    new_file_name, new_display_name = result
                    self.image_handler.rename_asset(old_name, new_display_name, is_mask)
                    self._update_asset_list()
                    self.statusBar().showMessage(f"Renamed mask to {new_display_name}")
                else:
                    QMessageBox.warning(self, "Rename Error", f"Could not rename mask: {result}")
            else:
                image_asset, _ = self.asset_manager.get_asset_pair(old_name)
                if image_asset:
                    image_asset.name = new_name
                    image_asset.save_project()
                    self.image_handler.rename_asset(old_name, new_name, is_mask)
                    self._update_asset_list()
                    self.statusBar().showMessage(f"Renamed image to {new_name}")
            self.cached_composite = None
            self._refresh_viewer()

    def _change_color(self, name, color_name, is_mask=False):
        if is_mask:
            for mask_asset in self.asset_manager.masks.values():
                if mask_asset.name == name:
                    mask_asset.pipeline.config["color"] = color_name
                    mask_asset.save_project()
                    break
        else:
            image_asset, _ = self.asset_manager.get_asset_pair(name)
            if image_asset:
                image_asset.pipeline.config["color"] = color_name
                image_asset.save_project()

        self.image_handler.set_asset_color(name, color_name)
        # Update tooltip in appropriate list
        list_widget = self.mask_list if is_mask else self.image_list
        for i in range(list_widget.count()):
            item = list_widget.item(i)
            if item.text() == name:
                item.setToolTip(f"Color: {color_name}")
                break
        self.cached_composite = None
        self._refresh_viewer()

    def _save_visible(self):
        if not self.image_handler.visible_assets and not self.image_handler.visible_masks:
            QMessageBox.warning(self, "No Assets Visible", "Please select at least one image or mask to save.")
            return

        dialog = SaveVisibleDialog(self, export_mode=None)
        if dialog.exec() != QDialog.Accepted:
            return

        options = dialog.get_options()
        output_dir = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if not output_dir:
            return

        try:
            saved_files = self.image_handler.save_visible(
                self.asset_manager,
                output_dir,
                options["filename"],
                options["format"]
            )
            
            if saved_files:
                QMessageBox.information(self, "Save Complete", f"Successfully saved visible view to {saved_files[0]}")
            else:
                QMessageBox.warning(self, "Save Failed", "No file was saved.")
        except Exception as e:
            QMessageBox.critical(self, "Error Saving View", f"An error occurred while saving: {str(e)}")

    def _export_modified_images(self):
        if not self.image_handler.visible_assets:
            QMessageBox.warning(self, "No Images Visible", "Please select at least one image to export.")
            return

        dialog = SaveVisibleDialog(self, export_mode='images')
        if dialog.exec() != QDialog.Accepted:
            return

        options = dialog.get_options()
        output_dir = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if not output_dir:
            return

        try:
            saved_files = self.image_handler.export_images(
                self.asset_manager,
                output_dir,
                options["filename"],
                options["format"]
            )
            
            if saved_files:
                QMessageBox.information(self, "Export Complete", f"Successfully exported {len(saved_files)} image(s) to {output_dir}")
            else:
                QMessageBox.warning(self, "Export Failed", "No images were exported.")
        except Exception as e:
            QMessageBox.critical(self, "Error Exporting Images", f"An error occurred while exporting: {str(e)}")

    def _export_modified_masks(self):
        if not self.image_handler.visible_masks:
            QMessageBox.warning(self, "No Masks Visible", "Please select at least one mask to export.")
            return

        dialog = SaveVisibleDialog(self, has_masks=len(self.image_handler.visible_masks) > 1, export_mode='masks')
        if dialog.exec() != QDialog.Accepted:
            return

        options = dialog.get_options()
        output_dir = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if not output_dir:
            return

        try:
            saved_files = self.image_handler.export_masks(
                self.asset_manager,
                output_dir,
                options["filename"],
                options["format"]
            )
            
            if saved_files:
                QMessageBox.information(self, "Export Complete", f"Successfully exported {len(saved_files)} mask(s) to {output_dir}")
            else:
                QMessageBox.warning(self, "Export Failed", "No masks were exported.")
        except Exception as e:
            QMessageBox.critical(self, "Error Exporting Masks", f"An error occurred while exporting: {str(e)}")

    def _select_all_images(self):
        for i in range(self.image_list.count()):
            item = self.image_list.item(i)
            name = item.text()
            if not self.image_handler.is_visible(name, is_mask=False):
                self.image_handler.toggle_visibility(name, is_mask=False)
            item.setSelected(True)
        self.cached_composite = None
        self._refresh_viewer()

    def _select_none_images(self):
        for i in range(self.image_list.count()):
            item = self.image_list.item(i)
            name = item.text()
            if self.image_handler.is_visible(name, is_mask=False):
                self.image_handler.toggle_visibility(name, is_mask=False)
            item.setSelected(False)
        self.cached_composite = None
        self._refresh_viewer()

    def _select_all_masks(self):
        for i in range(self.mask_list.count()):
            item = self.mask_list.item(i)
            name = item.text()
            if not self.image_handler.is_visible(name, is_mask=True):
                self.image_handler.toggle_visibility(name, is_mask=True)
            item.setSelected(True)
        self.cached_composite = None
        self._refresh_viewer()

    def _select_none_masks(self):
        for i in range(self.mask_list.count()):
            item = self.mask_list.item(i)
            name = item.text()
            if self.image_handler.is_visible(name, is_mask=True):
                self.image_handler.toggle_visibility(name, is_mask=True)
            item.setSelected(False)
        self.cached_composite = None
        self._refresh_viewer()

    def _asset_clicked(self, item):
        name = item.text()
        self.image_handler.toggle_visibility(name, is_mask=False)
        self.cached_composite = None # Invalidate cache
        
        # Visually reflect selection state
        item.setSelected(self.image_handler.is_visible(name, is_mask=False))
        
        self._refresh_viewer()

    def _mask_clicked(self, item):
        name = item.text()
        self.image_handler.toggle_visibility(name, is_mask=True)
        self.cached_composite = None # Invalidate cache
        
        # Visually reflect selection state
        item.setSelected(self.image_handler.is_visible(name, is_mask=True))
        
        self._refresh_viewer()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Delete:
            self._delete_selected_masks()
        super().keyPressEvent(event)

    def _delete_selected_masks(self):
        visible_masks = list(self.image_handler.visible_masks)
        if not visible_masks:
            return

        confirm = QMessageBox.question(self, "Confirm Delete", 
                                     f"Are you sure you want to delete {len(visible_masks)} selected mask(s)?\nThis cannot be undone.",
                                     QMessageBox.Yes | QMessageBox.No)
        
        if confirm == QMessageBox.Yes:
            for name in visible_masks:
                success, message = self.asset_manager.delete_mask(name)
                if success:
                    self.image_handler.remove_asset(name, is_mask=True)
                    print(f"Mask deleted: {name}")
                else:
                    QMessageBox.warning(self, "Delete Error", f"Failed to delete {name}: {message}")
            
            self.cached_composite = None
            self._update_asset_list()
            self._refresh_viewer()
            self.statusBar().showMessage(f"Deleted {len(visible_masks)} mask(s)")

    def _opacity_changed(self, value):
        opacity = value / 100.0
        self.image_handler.mask_opacity = opacity
        
        # Update current mask opacities in pipeline
        for name in self.image_handler.visible_masks:
            for mask_asset in self.asset_manager.masks.values():
                if mask_asset.name == name:
                    mask_asset.pipeline.config["opacity"] = opacity
                    mask_asset.save_project()
                    break

        self.cached_composite = None
        self._refresh_viewer()

    def _refresh_viewer(self):
        if self.cached_composite is None:
            composite_qimg = self.image_handler.render_composite(self.asset_manager)
            if composite_qimg:
                self.cached_composite = QPixmap.fromImage(composite_qimg)
            else:
                self.cached_composite = None

        self.viewer_view.set_pixmap(self.cached_composite)

    def _crop_all(self):
        # We need dimensions to show defaults. Let's pick the first image if any.
        img_names = self.asset_manager.get_image_list()
        if not img_names:
            QMessageBox.warning(self, "Crop", "No images loaded.")
            return

        # Check for selection in viewer
        selection = self.viewer_view.get_selection_rect()
        if selection:
            x, y, w, h = int(selection.x()), int(selection.y()), int(selection.width()), int(selection.height())
            
            confirm = QMessageBox.question(self, "Confirm Crop", 
                                         f"Crop all images and masks to selected area?\n(X: {x}, Y: {y}, W: {w}, H: {h})",
                                         QMessageBox.Yes | QMessageBox.No)
            if confirm == QMessageBox.Yes:
                self.asset_manager.apply_global_crop(x, y, w, h)
                self.viewer_view.clear_selection()
                self.cached_composite = None # Invalidate cache
                self._refresh_viewer()
                self._update_asset_list() # To update thumbnails
            return

        # Fallback to dialog if no selection
        # Get CURRENT size of first image to suggest crop
        image_asset, _ = self.asset_manager.get_asset_pair(img_names[0])
        data = image_asset.get_rendered_data(for_clustering=False)
        h_curr, w_curr = data.shape[:2]

        # Use a custom dialog.
        dialog = QDialog(self)
        dialog.setWindowTitle("Crop All Images and Masks")
        dialog_layout = QVBoxLayout(dialog)
        form_layout = QFormLayout()
        
        x_input = QSpinBox()
        x_input.setRange(0, w_curr-1)
        y_input = QSpinBox()
        y_input.setRange(0, h_curr-1)
        w_input = QSpinBox()
        w_input.setRange(1, w_curr)
        w_input.setValue(w_curr)
        h_input = QSpinBox()
        h_input.setRange(1, h_curr)
        h_input.setValue(h_curr)
        
        form_layout.addRow("X:", x_input)
        form_layout.addRow("Y:", y_input)
        form_layout.addRow("Width:", w_input)
        form_layout.addRow("Height:", h_input)
        dialog_layout.addLayout(form_layout)
        
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, dialog)
        dialog_layout.addWidget(buttons)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        
        if dialog.exec() == QDialog.Accepted:
            self.asset_manager.apply_global_crop(x_input.value(), y_input.value(), w_input.value(), h_input.value())
            self.cached_composite = None # Invalidate cache
            self._refresh_viewer()
            self._update_asset_list() # To update thumbnails

    def _rotate_all(self):
        angle, ok = QInputDialog.getDouble(self, "Rotate All", "Angle (degrees):", 0, -360, 360, 1)
        if ok:
            self.asset_manager.apply_global_rotate(angle)
            self.cached_composite = None # Invalidate cache
            self._refresh_viewer()
            self._update_asset_list()

    def _merge_masks_triggered(self):
        """Merges currently visible masks into a new mask."""
        visible_masks = sorted(list(self.image_handler.visible_masks))
        if len(visible_masks) < 2:
            QMessageBox.warning(self, "Selection Error", "Please toggle visibility for at least two masks to merge.")
            return

        new_mask_name, ok = QInputDialog.getText(self, "Merge Masks", "Enter name for new mask:", text="merged_mask")
        if not ok or not new_mask_name.strip():
            return

        new_mask_name = new_mask_name.strip()
        
        # Check if name exists
        if any(m.name == new_mask_name for m in self.asset_manager.masks.values()):
            QMessageBox.warning(self, "Invalid Name", f"A mask with name '{new_mask_name}' already exists.")
            return

        merged_data = None
        for name in visible_masks:
            mask_asset = None
            # Find the actual asset by its name or file_name
            if name in self.asset_manager.masks:
                mask_asset = self.asset_manager.masks[name]
            else:
                for m in self.asset_manager.masks.values():
                    if m.name == name:
                        mask_asset = m
                        break
            
            if not mask_asset:
                continue

            # Load mask data (binary: 0 or 255)
            data = mask_asset.data
            if data is None:
                continue

            if merged_data is None:
                merged_data = data.copy()
            else:
                if merged_data.shape != data.shape:
                    QMessageBox.warning(self, "Shape Mismatch", f"Mask '{name}' has a different shape {data.shape} than others {merged_data.shape}. Skipping.")
                    continue
                # Bitwise OR to merge masks
                merged_data = np.bitwise_or(merged_data, data)

        if merged_data is not None:
            # Add to asset manager
            self.asset_manager.add_new_mask(new_mask_name, merged_data)
            # Update UI
            self._update_asset_list()
            self.statusBar().showMessage(f"Merged masks into '{new_mask_name}'")
        else:
            QMessageBox.critical(self, "Merge Failed", "Could not load data for selected masks.")

    def _create_status_bar(self):
        self.statusBar().showMessage("Ready")
