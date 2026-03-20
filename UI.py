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
import json
from datetime import datetime
import tifffile
import numpy as np
from PIL import Image
from assets import AssetManager
from image_handler import ImageDisplayHandler
from graphs import calculate_normalized_graphs, create_graph_pixmap
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
            self.viewport().update()
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

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Clustering App")
        self.resize(1500, 1000)

        self.asset_manager = AssetManager()
        self.image_handler = ImageDisplayHandler()
        self.graphs = {} # name -> {image_data, mask_data, orig_image_name, orig_mask_name}
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

        # Graph list
        hist_container = QWidget()
        hist_h_layout = QHBoxLayout(hist_container)
        hist_h_layout.setContentsMargins(0, 0, 0, 0)
        
        hist_btn_layout = QVBoxLayout()
        hist_btn_layout.setContentsMargins(0, 0, 5, 0)
        
        self.select_all_hists_btn = QPushButton("Select\nAll")
        self.select_all_hists_btn.setFixedWidth(60)
        self.select_all_hists_btn.clicked.connect(self._select_all_graphs)
        self.select_none_hists_btn = QPushButton("Select\nNone")
        self.select_none_hists_btn.setFixedWidth(60)
        self.select_none_hists_btn.clicked.connect(self._select_none_graphs)
        
        hist_btn_layout.addWidget(self.select_all_hists_btn)
        hist_btn_layout.addWidget(self.select_none_hists_btn)
        hist_btn_layout.addStretch()
        
        self.graph_list = QListWidget()
        self.graph_list.setIconSize(QSize(100, 100))
        self.graph_list.setSelectionMode(QListWidget.MultiSelection)
        self.graph_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.graph_list.customContextMenuRequested.connect(self._show_graph_context_menu)
        self.graph_list.itemClicked.connect(self._graph_clicked)
        
        hist_h_layout.addLayout(hist_btn_layout)
        hist_h_layout.addWidget(self.graph_list)
        
        dock_layout.addWidget(QLabel("Graphs:"))
        dock_layout.addWidget(hist_container)

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
        self.viewer_subwindow.setAttribute(Qt.WA_DeleteOnClose, False)
        self.viewer_subwindow.setWindowTitle("Image Viewer")
        
        self.viewer_view = ZoomableView()
        
        self.viewer_subwindow.setWidget(self.viewer_view)
        self.mdi_area.addSubWindow(self.viewer_subwindow)
        self.viewer_subwindow.show()

        # Create the graph viewer in a sub-window
        self.graph_subwindow = QMdiSubWindow()
        self.graph_subwindow.setAttribute(Qt.WA_DeleteOnClose, False)
        self.graph_subwindow.setWindowTitle("Graph Viewer")
        self.graph_view = GraphViewer(self)
        self.graph_subwindow.setWidget(self.graph_view)
        self.mdi_area.addSubWindow(self.graph_subwindow)
        self.graph_subwindow.show()

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

        # "Analysis" dropdown button
        analysis_menu = menu_bar.addMenu("Analysis")

        graph_action = QAction("Create Graph", self)
        graph_action.triggered.connect(self._create_graph_triggered)
        analysis_menu.addAction(graph_action)

        bivariate_graph_action = QAction("Create Bivariate Graph", self)
        bivariate_graph_action.triggered.connect(self._create_bivariate_graph_triggered)
        analysis_menu.addAction(bivariate_graph_action)

        joint_kde_action = QAction("Create Joint KDE Graph", self)
        joint_kde_action.triggered.connect(self._create_joint_kde_triggered)
        analysis_menu.addAction(joint_kde_action)

        ridge_plot_action = QAction("Create Ridge Plot", self)
        ridge_plot_action.triggered.connect(self._create_ridgeplot_triggered)
        analysis_menu.addAction(ridge_plot_action)

        group_csv_action = QAction("Group .csv", self)
        group_csv_action.triggered.connect(self._save_group_graph_csv)
        analysis_menu.addAction(group_csv_action)

    def _home_triggered(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Working Directory")
        if directory:
            self.working_dir = directory
            self.asset_manager.set_working_dir(directory)
            
            # Load graphs if "Graphs" folder exists
            graph_dir = os.path.join(directory, "Graphs")
            if os.path.isdir(graph_dir):
                for filename in os.listdir(graph_dir):
                    if filename.endswith(".json"):
                        file_path = os.path.join(graph_dir, filename)
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                graph_data = json.load(f)
                                
                                # Could be a list of graphs or a single graph dictionary
                                if isinstance(graph_data, list):
                                    graph_list = graph_data
                                else:
                                    graph_list = [graph_data]
                                    
                                for graph_info in graph_list:
                                    name = graph_info.get("name")
                                    if not name:
                                        continue
                                        
                                    # To satisfy refresh_graph_view, we need image_data and mask_data, 
                                    # but we can also use pre-calculated data if those are missing.
                                    g_type = graph_info.get("type", "standard")
                                    
                                    # Initialize graph info
                                    self.graphs[name] = {'type': g_type}
                                    
                                    # Copy all available fields from JSON
                                    for key, value in graph_info.items():
                                        if key not in ['image_data', 'mask_data', 'image1_data', 'image2_data', 'image_datasets']:
                                            self.graphs[name][key] = value
                                    
                                    # Attempt to find the original images/masks in asset_manager
                                    if g_type == "standard":
                                        img_name = graph_info.get("image")
                                        mask_name = graph_info.get("mask")
                                        
                                        image_asset = self.asset_manager.get_image_by_name(img_name)
                                        mask_asset = self.asset_manager.get_mask_by_name(mask_name)
                                        
                                        if image_asset and mask_asset:
                                            self.graphs[name].update({
                                                'image_data': image_asset.get_rendered_data(for_clustering=False),
                                                'mask_data': mask_asset.data,
                                                'orig_image_name': img_name,
                                                'orig_mask_name': mask_name
                                            })
                                            # Default color? Let's use image color
                                            color = self.image_handler.get_asset_color(img_name)
                                            self.image_handler.set_asset_color(name, color)
                                        else:
                                            # If assets are missing, we still have name, type, counts, bins from JSON
                                            # Set a default color for missing assets
                                            self.image_handler.set_asset_color(name, "white")
                                            
                                    elif g_type in ["bivariate", "joint_kde"]:
                                        img1_name = graph_info.get("image1")
                                        img2_name = graph_info.get("image2")
                                        mask_name = graph_info.get("mask")
                                        
                                        image1_asset = self.asset_manager.get_image_by_name(img1_name)
                                        image2_asset = self.asset_manager.get_image_by_name(img2_name)
                                        mask_asset = self.asset_manager.get_mask_by_name(mask_name)
                                        
                                        if image1_asset and image2_asset and mask_asset:
                                            self.graphs[name].update({
                                                'image1_data': image1_asset.get_rendered_data(for_clustering=False),
                                                'image2_data': image2_asset.get_rendered_data(for_clustering=False),
                                                'mask_data': mask_asset.data,
                                                'image1_name': img1_name,
                                                'image2_name': img2_name,
                                                'orig_mask_name': mask_name
                                            })
                                        self.image_handler.set_asset_color(name, "magenta")
                                        
                                    elif g_type == "ridge":
                                        image_names = graph_info.get("image_names", [])
                                        mask_name = graph_info.get("mask")
                                        
                                        mask_asset = self.asset_manager.get_mask_by_name(mask_name)
                                        images_found = []
                                        for img_n in image_names:
                                            img_asset = self.asset_manager.get_image_by_name(img_n)
                                            if img_asset:
                                                images_found.append(img_asset.get_rendered_data(for_clustering=False))
                                            else:
                                                images_found.append(None)
                                                
                                        if mask_asset and all(img is not None for img in images_found):
                                            self.graphs[name].update({
                                                'image_datasets': images_found,
                                                'mask_data': mask_asset.data,
                                                'image_names': image_names,
                                                'orig_mask_name': mask_name
                                            })
                                        self.image_handler.set_asset_color(name, "cyan")

                        except Exception as e:
                            print(f"Error loading graph file {file_path}: {e}")

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
            mask_asset = self.asset_manager.get_mask_by_name(name)
            
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
            if self.image_handler.is_visible(name, is_graph=True):
                item.setSelected(True)

        for name in self.graphs.keys():
            item = QListWidgetItem(name)
            # We don't have thumbnails for graphs, but we could generate one or use a placeholder
            self.graph_list.addItem(item)
            if self.image_handler.is_visible(name, is_graph=True):
                item.setSelected(True)
            
            # If the graph has assets, they might already be visible or not.
            # We don't force selection here unless it was already visible in handler.

    def _show_image_context_menu(self, position: QPoint):
        self._show_context_menu(self.image_list, position, is_mask=False)

    def _show_mask_context_menu(self, position: QPoint):
        self._show_context_menu(self.mask_list, position, is_mask=True)

    def _show_graph_context_menu(self, position: QPoint):
        item = self.graph_list.itemAt(position)
        if not item:
            return

        name = item.text()
        menu = QMenu()
        
        # Rename
        rename_action = QAction("Rename", self)
        rename_action.triggered.connect(lambda: self._rename_graph(name))
        menu.addAction(rename_action)
        
        # Save CSV
        save_csv_action = QAction("Save .csv", self)
        save_csv_action.triggered.connect(lambda: self._save_graph_csv(name))
        menu.addAction(save_csv_action)
        
        menu.addSeparator()

        color_menu = menu.addMenu("Change Color")
        for color_name in self.image_handler.COLORS.keys():
            action = QAction(color_name.capitalize(), self)
            action.triggered.connect(lambda checked=False, n=name, c=color_name: self._change_color(n, c, is_graph=True))
            color_menu.addAction(action)

        menu.exec(self.graph_list.mapToGlobal(position))

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
                    self.image_handler.rename_asset(old_name, new_display_name, is_mask=True)
                    self._update_asset_list()
                    self.statusBar().showMessage(f"Renamed mask to {new_display_name}")
                else:
                    QMessageBox.warning(self, "Rename Error", f"Could not rename mask: {result}")
            else:
                image_asset, _ = self.asset_manager.get_asset_pair(old_name)
                if image_asset:
                    image_asset.name = new_name
                    image_asset.save_project()
                    self.image_handler.rename_asset(old_name, new_name, is_mask=False)
                    self._update_asset_list()
                    self.statusBar().showMessage(f"Renamed image to {new_name}")
            self.cached_composite = None
            self._refresh_viewer()

    def _change_color(self, name, color_name, is_mask=False, is_graph=False):
        if is_graph:
            pass # Graphs don't have persistent pipeline config yet
        elif is_mask:
            mask_asset = self.asset_manager.get_mask_by_name(name)
            if mask_asset:
                mask_asset.pipeline.config["color"] = color_name
                mask_asset.save_project()
        else:
            image_asset, _ = self.asset_manager.get_asset_pair(name)
            if image_asset:
                image_asset.pipeline.config["color"] = color_name
                image_asset.save_project()

        self.image_handler.set_asset_color(name, color_name)
        # Update tooltip in appropriate list
        if is_graph:
            list_widget = self.graph_list
        else:
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
        self.viewer_subwindow.show()
        self.viewer_subwindow.setFocus()
        self._refresh_viewer()

    def _select_none_images(self):
        for i in range(self.image_list.count()):
            item = self.image_list.item(i)
            name = item.text()
            if self.image_handler.is_visible(name, is_mask=False):
                self.image_handler.toggle_visibility(name, is_mask=False)
            item.setSelected(False)
        self.cached_composite = None
        self.viewer_subwindow.show()
        self.viewer_subwindow.setFocus()
        self._refresh_viewer()

    def _select_all_masks(self):
        for i in range(self.mask_list.count()):
            item = self.mask_list.item(i)
            name = item.text()
            if not self.image_handler.is_visible(name, is_mask=True):
                self.image_handler.toggle_visibility(name, is_mask=True)
            item.setSelected(True)
        self.cached_composite = None
        self.viewer_subwindow.show()
        self.viewer_subwindow.setFocus()
        self._refresh_viewer()

    def _select_none_masks(self):
        for i in range(self.mask_list.count()):
            item = self.mask_list.item(i)
            name = item.text()
            if self.image_handler.is_visible(name, is_mask=True):
                self.image_handler.toggle_visibility(name, is_mask=True)
            item.setSelected(False)
        self.cached_composite = None
        self.viewer_subwindow.show()
        self.viewer_subwindow.setFocus()
        self._refresh_viewer()

    def _asset_clicked(self, item):
        name = item.text()
        self.image_handler.toggle_visibility(name, is_mask=False)
        self.cached_composite = None # Invalidate cache
        
        # Visually reflect selection state
        item.setSelected(self.image_handler.is_visible(name, is_mask=False))
        
        self.viewer_subwindow.show()
        self.viewer_subwindow.setFocus()
        self._refresh_viewer()

    def _mask_clicked(self, item):
        name = item.text()
        self.image_handler.toggle_visibility(name, is_mask=True)
        self.cached_composite = None # Invalidate cache
        
        # Visually reflect selection state
        item.setSelected(self.image_handler.is_visible(name, is_mask=True))
        
        self.viewer_subwindow.show()
        self.viewer_subwindow.setFocus()
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
            mask_asset = self.asset_manager.get_mask_by_name(name)
            if mask_asset:
                mask_asset.pipeline.config["opacity"] = opacity
                mask_asset.save_project()

        self.cached_composite = None
        self._refresh_viewer()

    def _refresh_viewer(self):
        if self.cached_composite is None:
            composite_qimg = self.image_handler.render_composite(self.asset_manager, graphs=self.graphs)
            if composite_qimg:
                self.cached_composite = QPixmap.fromImage(composite_qimg)
            else:
                self.cached_composite = None

        self.viewer_view.set_pixmap(self.cached_composite)
        self.refresh_graph_view()

    def refresh_graph_view(self):
        if not hasattr(self, 'graph_view'):
            return

        visible_hists = sorted(list(self.image_handler.visible_graphs))
        if not visible_hists:
            self.graph_view.set_pixmap(None)
            return

        # Check for bivariate graphs first
        bivariate_hists = [name for name in visible_hists if name in self.graphs and self.graphs[name].get('type') == 'bivariate']
        
        if bivariate_hists:
            # We will show only the first selected bivariate graph for now
            name = bivariate_hists[0]
            hist_info = self.graphs[name]
            
            color_name = self.image_handler.get_asset_color(name)
            color_rgb = self.image_handler.COLORS.get(color_name, (1, 1, 1))
            from PySide6.QtGui import QColor
            color = QColor(int(color_rgb[0]*255), int(color_rgb[1]*255), int(color_rgb[2]*255))
            
            w = max(400, self.graph_view.width())
            h = max(300, self.graph_view.height())

            if 'image1_data' in hist_info and 'image2_data' in hist_info and 'mask_data' in hist_info:
                img1_data = hist_info['image1_data']
                img2_data = hist_info['image2_data']
                mask_data = hist_info['mask_data']
                
                # Apply mask to images
                m1 = np.zeros_like(img1_data)
                m2 = np.zeros_like(img2_data)
                m1[mask_data > 0] = img1_data[mask_data > 0]
                m2[mask_data > 0] = img2_data[mask_data > 0]
                
                from seaborn_utils import create_bivariate_kdeplot_pixmap
                names = [hist_info.get('image1_name', 'Img1'), hist_info.get('image2_name', 'Img2')]
                pixmap = create_bivariate_kdeplot_pixmap([m1, m2], [color, color], w, h, names=names)
                self.graph_view.set_pixmap(pixmap)
            elif 'image1_values' in hist_info and 'image2_values' in hist_info:
                # Use pre-calculated values
                from seaborn_utils import create_bivariate_kdeplot_pixmap
                # We need to simulate the masked images for the existing function 
                # or modify the function. Let's try to simulate.
                # Actually create_bivariate_kdeplot_pixmap calls get_bivariate_kde_data 
                # which does the sampling. If we already have samples...
                
                # I'll add a check in seaborn_utils.py to handle pre-sampled data too if possible.
                # For now, let's just use a placeholder or message if not easily possible.
                # Wait, I can just use the sampled values to "reconstruct" m1, m2 if I wanted, 
                # but better to update seaborn_utils.
                
                # Let's assume I'll update seaborn_utils.py to take these values.
                names = [hist_info.get('image1', 'Img1'), hist_info.get('image2', 'Img2')]
                # Pass values as a special dict
                pixmap = create_bivariate_kdeplot_pixmap(
                    {'x': hist_info['image1_values'], 'y': hist_info['image2_values']}, 
                    [color, color], w, h, names=names
                )
                self.graph_view.set_pixmap(pixmap)
            return

        # Check for joint KDE graphs
        joint_kde_hists = [name for name in visible_hists if name in self.graphs and self.graphs[name].get('type') == 'joint_kde']
        
        if joint_kde_hists:
            # Show only the first selected joint KDE graph
            name = joint_kde_hists[0]
            hist_info = self.graphs[name]
            
            color_name = self.image_handler.get_asset_color(name)
            color_rgb = self.image_handler.COLORS.get(color_name, (1, 1, 1))
            from PySide6.QtGui import QColor
            color = QColor(int(color_rgb[0]*255), int(color_rgb[1]*255), int(color_rgb[2]*255))
            
            w = max(400, self.graph_view.width())
            h = max(300, self.graph_view.height())

            if 'image1_data' in hist_info and 'image2_data' in hist_info and 'mask_data' in hist_info:
                img1_data = hist_info['image1_data']
                img2_data = hist_info['image2_data']
                mask_data = hist_info['mask_data']
                
                # Apply mask to images
                m1 = np.zeros_like(img1_data)
                m2 = np.zeros_like(img2_data)
                m1[mask_data > 0] = img1_data[mask_data > 0]
                m2[mask_data > 0] = img2_data[mask_data > 0]
                
                from seaborn_utils import create_joint_kdeplot_pixmap
                names = [hist_info.get('image1_name', 'Img1'), hist_info.get('image2_name', 'Img2')]
                pixmap = create_joint_kdeplot_pixmap([m1, m2], [color, color], w, h, names=names)
                self.graph_view.set_pixmap(pixmap)
            elif 'image1_values' in hist_info and 'image2_values' in hist_info:
                from seaborn_utils import create_joint_kdeplot_pixmap
                names = [hist_info.get('image1', 'Img1'), hist_info.get('image2', 'Img2')]
                pixmap = create_joint_kdeplot_pixmap(
                    {'x': hist_info['image1_values'], 'y': hist_info['image2_values']}, 
                    [color, color], w, h, names=names
                )
                self.graph_view.set_pixmap(pixmap)
            return

        # Check for ridge plots
        ridge_hists = [name for name in visible_hists if name in self.graphs and self.graphs[name].get('type') == 'ridge']

        if ridge_hists:
            # Show only the first selected ridge plot
            name = ridge_hists[0]
            hist_info = self.graphs[name]
            
            from PySide6.QtGui import QColor
            w = max(400, self.graph_view.width())
            h = max(300, self.graph_view.height())
            from seaborn_utils import create_ridgeplot_pixmap

            if 'image_datasets' in hist_info and 'mask_data' in hist_info:
                image_datasets = hist_info['image_datasets']
                mask_data = hist_info['mask_data']
                image_names = hist_info['image_names']

                masked_images = []
                colors = []
                for i, img_data in enumerate(image_datasets):
                    m = np.zeros_like(img_data)
                    m[mask_data > 0] = img_data[mask_data > 0]
                    masked_images.append(m)
                    
                    # Try to get color for original image, else use graph color
                    img_name = image_names[i]
                    color_name = self.image_handler.get_asset_color(img_name)
                    color_rgb = self.image_handler.COLORS.get(color_name, (1, 1, 1))
                    colors.append(QColor(int(color_rgb[0]*255), int(color_rgb[1]*255), int(color_rgb[2]*255)))

                pixmap = create_ridgeplot_pixmap(masked_images, colors, w, h, names=image_names)
                self.graph_view.set_pixmap(pixmap)
            elif 'image_datasets_values' in hist_info:
                # Use pre-calculated values for each image in the ridge plot
                image_names = hist_info.get('image_names', [])
                datasets_values = hist_info['image_datasets_values'] # list of lists
                
                colors = []
                for img_name in image_names:
                    color_name = self.image_handler.get_asset_color(img_name)
                    color_rgb = self.image_handler.COLORS.get(color_name, (1, 1, 1))
                    colors.append(QColor(int(color_rgb[0]*255), int(color_rgb[1]*255), int(color_rgb[2]*255)))
                
                # We need to update create_ridgeplot_pixmap to handle pre-sampled values
                pixmap = create_ridgeplot_pixmap(datasets_values, colors, w, h, names=image_names)
                self.graph_view.set_pixmap(pixmap)
            return

        datasets = []
        colors = []
        labels = []
        
        from PySide6.QtGui import QColor
        
        for name in visible_hists:
            if name in self.graphs:
                hist_info = self.graphs[name]
                if hist_info.get('type') in ['bivariate', 'joint_kde', 'ridge']:
                    continue # Already handled or skipped if multiple
                
                if 'image_data' in hist_info and 'mask_data' in hist_info:
                    img_data = hist_info['image_data']
                    mask_data = hist_info['mask_data']
                    # Apply mask and flatten
                    data = img_data[mask_data > 0].flatten()
                    datasets.append(data)
                elif 'counts' in hist_info and 'bins' in hist_info:
                    # Use pre-calculated data
                    datasets.append({
                        'counts': hist_info['counts'],
                        'bins': hist_info['bins']
                    })
                else:
                    continue
                
                color_name = self.image_handler.get_asset_color(name)
                color_rgb = self.image_handler.COLORS.get(color_name, (1, 1, 1))
                colors.append(QColor(int(color_rgb[0]*255), int(color_rgb[1]*255), int(color_rgb[2]*255)))
                labels.append(name)

        if datasets:
            from seaborn_utils import create_multi_graph_plot_pixmap
            # Use current widget size for plotting
            w = max(400, self.graph_view.width())
            h = max(300, self.graph_view.height())
            pixmap = create_multi_graph_plot_pixmap(datasets, colors, labels, w, h)
            self.graph_view.set_pixmap(pixmap)
        else:
            self.graph_view.set_pixmap(None)

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
            mask_asset = self.asset_manager.get_mask_by_name(name)
            
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

    def _create_graph_triggered(self):
        """Creates graphs for selected images based on one selected mask."""
        selected_images = sorted(list(self.image_handler.visible_assets))
        selected_masks = sorted(list(self.image_handler.visible_masks))

        if not selected_images:
            QMessageBox.warning(self, "Selection Error", "Please select at least one image.")
            return

        if len(selected_masks) != 1:
            QMessageBox.warning(self, "Selection Error", "Please select exactly one mask to use as ROI.")
            return

        mask_name = selected_masks[0]
        mask_asset = self.asset_manager.get_mask_by_name(mask_name)
        
        if not mask_asset:
            QMessageBox.critical(self, "Error", f"Could not find mask asset for '{mask_name}'")
            return

        mask_data = mask_asset.data
        if mask_data is None:
            QMessageBox.critical(self, "Error", f"Could not load data for mask '{mask_name}'")
            return

        new_hists = []
        for img_name in selected_images:
            image_asset, _ = self.asset_manager.get_asset_pair(img_name)
            if not image_asset:
                continue
            
            image_data = image_asset.get_rendered_data(for_clustering=False)
            if image_data is None:
                continue

            # Ensure shapes match
            if image_data.shape[:2] != mask_data.shape[:2]:
                QMessageBox.warning(self, "Shape Mismatch", f"Image '{img_name}' shape {image_data.shape[:2]} does not match mask '{mask_name}' shape {mask_data.shape[:2]}. Skipping.")
                continue

            hist_name = f"Hist_{img_name}_{mask_name}"
            # Ensure unique name
            base_hist_name = hist_name
            counter = 1
            while hist_name in self.graphs:
                hist_name = f"{base_hist_name}_{counter}"
                counter += 1

            self.graphs[hist_name] = {
                'image_data': image_data,
                'mask_data': mask_data,
                'orig_image_name': img_name,
                'orig_mask_name': mask_name
            }
            # Set default color to match image if possible, or just magenta
            color = self.image_handler.get_asset_color(img_name)
            self.image_handler.set_asset_color(hist_name, color)
            self.image_handler.toggle_visibility(hist_name, is_graph=True)
            new_hists.append(hist_name)

        if new_hists:
            # Export new graphs as JSON
            export_data = []
            for h_name in new_hists:
                h_info = self.graphs[h_name]
                img_data = h_info['image_data']
                msk_data = h_info['mask_data']
                
                # Apply mask and calculate histogram
                masked_pixels = img_data[msk_data > 0].flatten()
                if len(masked_pixels) > 0:
                    counts, bin_edges = np.histogram(masked_pixels, bins=256, range=(0, 256))
                    export_data.append({
                        "name": h_name,
                        "image": h_info['orig_image_name'],
                        "mask": h_info['orig_mask_name'],
                        "counts": counts.tolist(),
                        "bins": bin_edges.tolist()
                    })

            if export_data:
                if self.working_dir:
                    hist_dir = os.path.join(self.working_dir, "Graphs")
                else:
                    hist_dir = "Graphs"
                    
                if not os.path.exists(hist_dir):
                    os.makedirs(hist_dir)
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"graphs_{timestamp}.json"
                file_path = os.path.join(hist_dir, filename)
                
                with open(file_path, 'w') as f:
                    json.dump(export_data, f, indent=4)
                
                print(f"Exported {len(export_data)} graphs to {file_path}")

            self._update_asset_list()
            self.cached_composite = None
            self.graph_subwindow.show()
            self.graph_subwindow.setFocus()
            self._refresh_viewer()
            self.statusBar().showMessage(f"Created {len(new_hists)} graph(s)")
        else:
            QMessageBox.warning(self, "No Graphs Created", "No graphs were created. Check image and mask selection.")

    def _create_bivariate_graph_triggered(self):
        """Creates a bivariate graph entry for two selected images based on one selected mask."""
        selected_images = sorted(list(self.image_handler.visible_assets))
        selected_masks = sorted(list(self.image_handler.visible_masks))

        if len(selected_images) != 2:
            QMessageBox.warning(self, "Selection Error", "Please select exactly two images for a bivariate graph.")
            return

        if len(selected_masks) != 1:
            QMessageBox.warning(self, "Selection Error", "Please select exactly one mask to use as ROI.")
            return

        mask_name = selected_masks[0]
        mask_asset = self.asset_manager.get_mask_by_name(mask_name)
        
        if not mask_asset:
            QMessageBox.critical(self, "Error", f"Could not find mask asset for '{mask_name}'")
            return

        mask_data = mask_asset.data
        if mask_data is None:
            QMessageBox.critical(self, "Error", f"Could not load data for mask '{mask_name}'")
            return

        img1_name = selected_images[0]
        img2_name = selected_images[1]
        
        image1_asset, _ = self.asset_manager.get_asset_pair(img1_name)
        image2_asset, _ = self.asset_manager.get_asset_pair(img2_name)
        
        if not image1_asset or not image2_asset:
            QMessageBox.critical(self, "Error", "Could not find asset data for selected images.")
            return
            
        img1_data = image1_asset.get_rendered_data(for_clustering=False)
        img2_data = image2_asset.get_rendered_data(for_clustering=False)
        
        if img1_data is None or img2_data is None:
            QMessageBox.critical(self, "Error", "Could not load data for selected images.")
            return

        # Ensure shapes match
        if img1_data.shape[:2] != mask_data.shape[:2] or img2_data.shape[:2] != mask_data.shape[:2]:
            QMessageBox.warning(self, "Shape Mismatch", f"Image shapes do not match mask '{mask_name}' shape {mask_data.shape[:2]}.")
            return

        hist_name = f"Bivariate_{img1_name}_{img2_name}_{mask_name}"
        # Ensure unique name
        base_hist_name = hist_name
        counter = 1
        while hist_name in self.graphs:
            hist_name = f"{base_hist_name}_{counter}"
            counter += 1

        self.graphs[hist_name] = {
            'type': 'bivariate',
            'image1_data': img1_data,
            'image2_data': img2_data,
            'mask_data': mask_data,
            'image1_name': img1_name,
            'image2_name': img2_name,
            'orig_mask_name': mask_name
        }
        
        self.image_handler.set_asset_color(hist_name, "magenta")
        self.image_handler.toggle_visibility(hist_name, is_graph=True)
        
        # Export bivariate graph as JSON
        export_data = {
            "name": hist_name,
            "type": "bivariate",
            "image1": img1_name,
            "image2": img2_name,
            "mask": mask_name,
            "timestamp": datetime.now().isoformat()
        }
        
        # Get sampled data points for the bivariate plot
        from seaborn_utils import get_bivariate_kde_data
        
        # Apply mask
        m1 = np.zeros_like(img1_data)
        m2 = np.zeros_like(img2_data)
        m1[mask_data > 0] = img1_data[mask_data > 0]
        m2[mask_data > 0] = img2_data[mask_data > 0]
        
        kde_data = get_bivariate_kde_data([m1, m2])
        if kde_data:
            x_vals, y_vals = kde_data
            export_data["image1_values"] = x_vals.tolist()
            export_data["image2_values"] = y_vals.tolist()
            export_data["sample_size"] = len(x_vals)

            if self.working_dir:
                hist_dir = os.path.join(self.working_dir, "Graphs")
            else:
                hist_dir = "Graphs"
                
            if not os.path.exists(hist_dir):
                os.makedirs(hist_dir)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"bivariate_graph_{timestamp}.json"
            file_path = os.path.join(hist_dir, filename)
            
            with open(file_path, 'w') as f:
                json.dump(export_data, f, indent=4)
            
            print(f"Exported bivariate graph to {file_path}")

        self._update_asset_list()
        self.cached_composite = None
        self.graph_subwindow.show()
        self.graph_subwindow.setFocus()
        self._refresh_viewer()
        self.statusBar().showMessage(f"Created bivariate graph '{hist_name}'")

    def _create_joint_kde_triggered(self):
        """Creates a joint KDE graph entry for two selected images based on one selected mask."""
        selected_images = sorted(list(self.image_handler.visible_assets))
        selected_masks = sorted(list(self.image_handler.visible_masks))

        if len(selected_images) != 2:
            QMessageBox.warning(self, "Selection Error", "Please select exactly two images for a joint KDE graph.")
            return

        if len(selected_masks) != 1:
            QMessageBox.warning(self, "Selection Error", "Please select exactly one mask to use as ROI.")
            return

        mask_name = selected_masks[0]
        mask_asset = self.asset_manager.get_mask_by_name(mask_name)
        
        if not mask_asset:
            QMessageBox.critical(self, "Error", f"Could not find mask asset for '{mask_name}'")
            return

        mask_data = mask_asset.data
        if mask_data is None:
            QMessageBox.critical(self, "Error", f"Could not load data for mask '{mask_name}'")
            return

        img1_name = selected_images[0]
        img2_name = selected_images[1]
        
        image1_asset, _ = self.asset_manager.get_asset_pair(img1_name)
        image2_asset, _ = self.asset_manager.get_asset_pair(img2_name)
        
        if not image1_asset or not image2_asset:
            QMessageBox.critical(self, "Error", "Could not find asset data for selected images.")
            return
            
        img1_data = image1_asset.get_rendered_data(for_clustering=False)
        img2_data = image2_asset.get_rendered_data(for_clustering=False)
        
        if img1_data is None or img2_data is None:
            QMessageBox.critical(self, "Error", "Could not load data for selected images.")
            return

        # Ensure shapes match
        if img1_data.shape[:2] != mask_data.shape[:2] or img2_data.shape[:2] != mask_data.shape[:2]:
            QMessageBox.warning(self, "Shape Mismatch", f"Image shapes do not match mask '{mask_name}' shape {mask_data.shape[:2]}.")
            return

        hist_name = f"JointKDE_{img1_name}_{img2_name}_{mask_name}"
        # Ensure unique name
        base_hist_name = hist_name
        counter = 1
        while hist_name in self.graphs:
            hist_name = f"{base_hist_name}_{counter}"
            counter += 1

        self.graphs[hist_name] = {
            'type': 'joint_kde',
            'image1_data': img1_data,
            'image2_data': img2_data,
            'mask_data': mask_data,
            'image1_name': img1_name,
            'image2_name': img2_name,
            'orig_mask_name': mask_name
        }
        
        self.image_handler.set_asset_color(hist_name, "cyan")
        self.image_handler.toggle_visibility(hist_name, is_graph=True)
        
        # Export joint KDE graph as JSON
        export_data = {
            "name": hist_name,
            "type": "joint_kde",
            "image1": img1_name,
            "image2": img2_name,
            "mask": mask_name,
            "timestamp": datetime.now().isoformat()
        }
        
        # Get sampled data points for the joint KDE plot
        from seaborn_utils import get_bivariate_kde_data
        
        # Apply mask
        m1 = np.zeros_like(img1_data)
        m2 = np.zeros_like(img2_data)
        m1[mask_data > 0] = img1_data[mask_data > 0]
        m2[mask_data > 0] = img2_data[mask_data > 0]
        
        kde_data = get_bivariate_kde_data([m1, m2])
        if kde_data:
            x_vals, y_vals = kde_data
            export_data["image1_values"] = x_vals.tolist()
            export_data["image2_values"] = y_vals.tolist()
            export_data["sample_size"] = len(x_vals)

            if self.working_dir:
                hist_dir = os.path.join(self.working_dir, "Graphs")
            else:
                hist_dir = "Graphs"
                
            if not os.path.exists(hist_dir):
                os.makedirs(hist_dir)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"joint_kde_graph_{timestamp}.json"
            file_path = os.path.join(hist_dir, filename)
            
            with open(file_path, 'w') as f:
                json.dump(export_data, f, indent=4)
            
            print(f"Exported joint KDE graph to {file_path}")

        self._update_asset_list()
        self.cached_composite = None
        self.graph_subwindow.show()
        self.graph_subwindow.setFocus()
        self._refresh_viewer()
        self.statusBar().showMessage(f"Created Joint KDE graph")

    def _create_ridgeplot_triggered(self):
        """Creates a ridge plot entry for selected images based on one selected mask."""
        selected_images = sorted(list(self.image_handler.visible_assets))
        selected_masks = sorted(list(self.image_handler.visible_masks))

        if len(selected_images) < 2:
            QMessageBox.warning(self, "Selection Error", "Please select at least two images for a ridge plot.")
            return

        if len(selected_masks) != 1:
            QMessageBox.warning(self, "Selection Error", "Please select exactly one mask to use as ROI.")
            return

        mask_name = selected_masks[0]
        mask_asset = self.asset_manager.get_mask_by_name(mask_name)
        
        if not mask_asset:
            QMessageBox.critical(self, "Error", f"Could not find mask asset for '{mask_name}'")
            return

        mask_data = mask_asset.data
        if mask_data is None:
            QMessageBox.critical(self, "Error", f"Could not load data for mask '{mask_name}'")
            return

        image_datasets = []
        image_names = []
        
        for img_name in selected_images:
            image_asset, _ = self.asset_manager.get_asset_pair(img_name)
            if not image_asset:
                continue
            
            img_data = image_asset.get_rendered_data(for_clustering=False)
            if img_data is None:
                continue
                
            if img_data.shape[:2] != mask_data.shape[:2]:
                QMessageBox.warning(self, "Shape Mismatch", f"Image '{img_name}' shape does not match mask shape. Skipping.")
                continue
                
            image_datasets.append(img_data)
            image_names.append(img_name)

        if len(image_datasets) < 2:
            QMessageBox.warning(self, "Selection Error", "Not enough images with matching shapes were found.")
            return

        hist_name = f"Ridge_{'_'.join(image_names)}_{mask_name}"
        if len(hist_name) > 100: # Truncate if too long
             hist_name = f"Ridge_{len(image_names)}imgs_{mask_name}"
             
        # Ensure unique name
        base_hist_name = hist_name
        counter = 1
        while hist_name in self.graphs:
            hist_name = f"{base_hist_name}_{counter}"
            counter += 1

        self.graphs[hist_name] = {
            'type': 'ridge',
            'image_datasets': image_datasets,
            'mask_data': mask_data,
            'image_names': image_names,
            'orig_mask_name': mask_name
        }
        
        self.image_handler.set_asset_color(hist_name, "magenta")
        self.image_handler.toggle_visibility(hist_name, is_graph=True)
        
        # Export ridge plot as JSON
        export_data = {
            "name": hist_name,
            "type": "ridge",
            "images": image_names,
            "mask": mask_name,
            "timestamp": datetime.now().isoformat(),
            "datasets": []
        }
        
        for i, img_data in enumerate(image_datasets):
            masked_pixels = img_data[mask_data > 0].flatten()
            if len(masked_pixels) > 0:
                if len(masked_pixels) > 10000:
                    indices = np.random.choice(len(masked_pixels), 10000, replace=False)
                    sampled_pixels = masked_pixels[indices]
                else:
                    sampled_pixels = masked_pixels
                    
                export_data["datasets"].append({
                    "image": image_names[i],
                    "values": sampled_pixels.tolist()
                })

        if self.working_dir:
            hist_dir = os.path.join(self.working_dir, "Graphs")
        else:
            hist_dir = "Graphs"
            
        if not os.path.exists(hist_dir):
            os.makedirs(hist_dir)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"ridge_plot_{timestamp}.json"
        file_path = os.path.join(hist_dir, filename)
        
        with open(file_path, 'w') as f:
            json.dump(export_data, f, indent=4)
        
        print(f"Exported ridge plot to {file_path}")

        self._update_asset_list()
        self.cached_composite = None
        self.graph_subwindow.show()
        self.graph_subwindow.setFocus()
        self._refresh_viewer()
        self.statusBar().showMessage(f"Created Ridge Plot")

    def _rename_graph(self, old_name):
        new_name, ok = QInputDialog.getText(self, "Rename Graph", "Enter new name:", text=old_name)
        if ok and new_name and new_name != old_name:
            if new_name in self.graphs:
                QMessageBox.warning(self, "Invalid Name", f"A graph with name '{new_name}' already exists.")
                return
            
            self.graphs[new_name] = self.graphs.pop(old_name)
            self.image_handler.rename_asset(old_name, new_name, is_graph=True)
            self._update_asset_list()
            self.statusBar().showMessage(f"Renamed graph to {new_name}")
            self.cached_composite = None
            self._refresh_viewer()

    def _save_graph_csv(self, name):
        if name not in self.graphs:
            return
            
        hist_info = self.graphs[name]
        
        if hist_info.get('type') == 'bivariate':
            # Bivariate data export to CSV - maybe as X, Y points?
            # Or just warn that it's not supported for bivariate yet
            QMessageBox.information(self, "Not Supported", "CSV export for bivariate graphs is not yet implemented.")
            return

        image_data = hist_info['image_data']
        mask_data = hist_info['mask_data']
        
        # Apply mask
        data = image_data[mask_data > 0].flatten()
        
        if len(data) == 0:
            QMessageBox.warning(self, "No Data", "No data points found within the mask for this graph.")
            return

        # Calculate histogram
        hist, bins = np.histogram(data, bins=256, range=(0, 256))
        
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Graph CSV", f"{name}.csv", "CSV Files (*.csv)")
        if file_path:
            try:
                import pandas as pd
                df = pd.DataFrame({
                    'Intensity': bins[:-1],
                    'Count': hist
                })
                df.to_csv(file_path, index=False)
                QMessageBox.information(self, "Save Complete", f"Successfully saved graph data to {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Save Error", f"An error occurred while saving CSV: {str(e)}")

    def _save_group_graph_csv(self):
        """Exports all visible graphs into a single .csv file."""
        visible_hists = sorted(list(self.image_handler.visible_graphs))
        if not visible_hists:
            QMessageBox.warning(self, "No Graphs Visible", "Please toggle visibility for at least one graph to export.")
            return

        file_path, _ = QFileDialog.getSaveFileName(self, "Save Group Graph CSV", "grouped_graphs.csv", "CSV Files (*.csv)")
        if not file_path:
            return

        try:
            import pandas as pd
            # Use dictionary to build DataFrame
            data_dict = {'Intensity': np.arange(256)}
            
            for name in visible_hists:
                if name not in self.graphs:
                    continue
                
                hist_info = self.graphs[name]
                if hist_info.get('type') == 'bivariate':
                    continue # Skip bivariate in grouped CSV for now
                
                image_data = hist_info['image_data']
                mask_data = hist_info['mask_data']
                
                # Apply mask to get data
                data = image_data[mask_data > 0].flatten()
                
                if len(data) == 0:
                    hist = np.zeros(256, dtype=int)
                else:
                    # Calculate histogram with 256 bins for intensities 0-255
                    hist, _ = np.histogram(data, bins=256, range=(0, 256))
                
                data_dict[name] = hist
                
            df = pd.DataFrame(data_dict)
            df.to_csv(file_path, index=False)
            QMessageBox.information(self, "Save Complete", f"Successfully saved grouped graph data to {file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"An error occurred while saving grouped CSV: {str(e)}")

    def _graph_clicked(self, item):
        name = item.text()
        self.image_handler.toggle_visibility(name, is_graph=True)
        self.cached_composite = None
        self.graph_subwindow.show()
        self.graph_subwindow.setFocus()
        self._refresh_viewer()

    def _select_all_graphs(self):
        for i in range(self.graph_list.count()):
            item = self.graph_list.item(i)
            name = item.text()
            if not self.image_handler.is_visible(name, is_graph=True):
                self.image_handler.toggle_visibility(name, is_graph=True)
            item.setSelected(True)
        self.cached_composite = None
        self.graph_subwindow.show()
        self.graph_subwindow.setFocus()
        self._refresh_viewer()

    def _select_none_graphs(self):
        for i in range(self.graph_list.count()):
            item = self.graph_list.item(i)
            name = item.text()
            if self.image_handler.is_visible(name, is_graph=True):
                self.image_handler.toggle_visibility(name, is_graph=True)
            item.setSelected(False)
        self.cached_composite = None
        self.graph_subwindow.show()
        self.graph_subwindow.setFocus()
        self._refresh_viewer()

    def _create_status_bar(self):
        self.statusBar().showMessage("Ready")

class GraphViewer(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setRenderHint(QPainter.Antialiasing)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setBackgroundRole(QPalette.NoRole)
        self.setFrameShape(QFrame.NoFrame)

        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)

        self.pixmap_item = QGraphicsPixmapItem()
        self.scene.addItem(self.pixmap_item)

    def set_pixmap(self, pixmap):
        if pixmap:
            self.pixmap_item.setPixmap(pixmap)
            self.scene.setSceneRect(QRectF(pixmap.rect()))
            self.viewport().update()
        else:
            self.pixmap_item.setPixmap(QPixmap())
            self.scene.setSceneRect(QRectF())

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # We might want to trigger a re-render here if we want the plot to scale with window
        # For now, just call parent.refresh_graph_view() if parent has it
        parent = self.parent()
        while parent and not hasattr(parent, 'refresh_graph_view'):
            parent = parent.parent()
        if parent:
            parent.refresh_graph_view()
