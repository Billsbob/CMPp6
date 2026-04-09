from PySide6.QtWidgets import (
    QMainWindow, QMenu, QMenuBar, QFileDialog, QListWidget, QListWidgetItem, 
    QWidget, QVBoxLayout, QLabel, QMdiArea, QMdiSubWindow, QDockWidget,
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QGraphicsRectItem, QFrame, 
    QMessageBox, QPushButton, QHBoxLayout, QDialog, QFormLayout, QDoubleSpinBox, 
    QSpinBox, QDialogButtonBox, QLineEdit, QComboBox, QInputDialog, QSlider,
    QApplication, QCheckBox
)
from PySide6.QtGui import QAction, QPixmap, QPainter, QWheelEvent, QPalette, QPen, QColor, QBrush, QImage
from PySide6.QtCore import Qt, QSize, QPoint, QPointF, QRectF, Signal
import os
import numpy as np
from PIL import Image
import json
from assets import AssetManager
from image_handler import ImageDisplayHandler
import image_stacker
import clustering
import graphing

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
            if isinstance(pixmap, QImage):
                pixmap = QPixmap.fromImage(pixmap)
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
                pen.setCosmetic(True)
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
                new_top_left = current_scene_pos - self.move_offset
                rect = self.selection_rect_item.rect()
                rect.moveTo(new_top_left)
                
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
                rect = QRectF(self.start_scene_pos, current_scene_pos).normalized()
                rect = rect.intersected(img_rect)
                self.selection_rect_item.setRect(rect)
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.start_scene_pos = None
            self.moving_selection = False
        super().mouseReleaseEvent(event)

    def get_selection_rect(self):
        if self.selection_rect_item:
            rect = self.selection_rect_item.rect()
            return (int(rect.x()), int(rect.y()), int(rect.width()), int(rect.height()))
        return None

    def wheelEvent(self, event: QWheelEvent):
        factor = 1.15
        if event.angleDelta().y() < 0:
            factor = 1.0 / factor
        
        if event.modifiers() == Qt.ControlModifier:
            self.scale(factor, factor)
            self.zoom_factor *= factor
        else:
            super().wheelEvent(event)

class FilterParameterDialog(QDialog):
    def __init__(self, filter_name, initial_params, parent=None):
        super().__init__(parent)
        self.filter_name = filter_name
        self.params = initial_params.copy()
        self.setup_ui()

    def setup_ui(self):
        self.setWindowTitle(f"{self.filter_name.capitalize()} Parameters")
        layout = QVBoxLayout(self)
        form_layout = QFormLayout()

        if self.filter_name == "gaussian":
            self.radius_spin = QDoubleSpinBox()
            self.radius_spin.setRange(0.1, 100.0)
            self.radius_spin.setValue(self.params.get("radius", 2.0))
            self.radius_spin.valueChanged.connect(lambda v: self._update_param("radius", v))
            form_layout.addRow("Radius:", self.radius_spin)
        elif self.filter_name in ["median", "mean", "blur"]:
            self.size_spin = QSpinBox()
            self.size_spin.setRange(1, 101)
            self.size_spin.setSingleStep(2)
            self.size_spin.setValue(self.params.get("size", 3))
            self.size_spin.valueChanged.connect(lambda v: self._update_param("size", v))
            form_layout.addRow("Size:", self.size_spin)
        elif self.filter_name == "unsharp":
            self.radius_spin = QDoubleSpinBox()
            self.radius_spin.setValue(self.params.get("radius", 2.0))
            self.radius_spin.valueChanged.connect(lambda v: self._update_param("radius", v))
            form_layout.addRow("Radius:", self.radius_spin)
            
            self.percent_spin = QSpinBox()
            self.percent_spin.setRange(1, 1000)
            self.percent_spin.setValue(self.params.get("percent", 150))
            self.percent_spin.valueChanged.connect(lambda v: self._update_param("percent", v))
            form_layout.addRow("Percent:", self.percent_spin)
            
            self.threshold_spin = QSpinBox()
            self.threshold_spin.setRange(0, 255)
            self.threshold_spin.setValue(self.params.get("threshold", 3))
            self.threshold_spin.valueChanged.connect(lambda v: self._update_param("threshold", v))
            form_layout.addRow("Threshold:", self.threshold_spin)

        layout.addLayout(form_layout)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _update_param(self, key, value):
        self.params[key] = value

    def get_params(self):
        return self.params

class ClusterParameterDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.params = {
            "n_clusters": 8,
            "max_iter": 300,
            "init": "k-means++",
            "n_init": 10,
            "random_state": 0,
            "algorithm": "auto",
            "normalize": False,
            "normalize_stack": False,
            "tol": 1e-4
        }
        self.setup_ui()

    def setup_ui(self):
        self.setWindowTitle("K-Means Parameters")
        layout = QVBoxLayout(self)
        form_layout = QFormLayout()

        self.n_clusters_spin = QSpinBox()
        self.n_clusters_spin.setRange(2, 100)
        self.n_clusters_spin.setValue(self.params["n_clusters"])
        form_layout.addRow("Number of Clusters:", self.n_clusters_spin)

        self.max_iter_spin = QSpinBox()
        self.max_iter_spin.setRange(1, 10000)
        self.max_iter_spin.setValue(self.params["max_iter"])
        form_layout.addRow("Max Iterations:", self.max_iter_spin)

        self.init_combo = QComboBox()
        self.init_combo.addItems(["k-means++", "random"])
        self.init_combo.setCurrentText(self.params["init"])
        form_layout.addRow("Init:", self.init_combo)

        self.n_init_spin = QSpinBox()
        self.n_init_spin.setRange(1, 100)
        self.n_init_spin.setValue(self.params["n_init"])
        form_layout.addRow("N Init:", self.n_init_spin)

        self.random_state_spin = QSpinBox()
        self.random_state_spin.setRange(0, 1000000)
        self.random_state_spin.setValue(self.params["random_state"])
        form_layout.addRow("Random State:", self.random_state_spin)

        self.algorithm_combo = QComboBox()
        self.algorithm_combo.addItems(["auto", "lloyd", "elkan", "full"])
        self.algorithm_combo.setCurrentText(self.params["algorithm"])
        form_layout.addRow("Algorithm:", self.algorithm_combo)

        self.normalize_check = QCheckBox()
        self.normalize_check.setChecked(self.params["normalize"])
        form_layout.addRow("Normalize (per image):", self.normalize_check)

        self.normalize_stack_check = QCheckBox()
        self.normalize_stack_check.setChecked(self.params["normalize_stack"])
        form_layout.addRow("Normalize (entire stack):", self.normalize_stack_check)

        self.tol_spin = QDoubleSpinBox()
        self.tol_spin.setRange(0, 1)
        self.tol_spin.setDecimals(6)
        self.tol_spin.setSingleStep(0.0001)
        self.tol_spin.setValue(self.params["tol"])
        form_layout.addRow("Tolerance:", self.tol_spin)

        layout.addLayout(form_layout)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_params(self):
        return {
            "n_clusters": self.n_clusters_spin.value(),
            "max_iter": self.max_iter_spin.value(),
            "init": self.init_combo.currentText(),
            "n_init": self.n_init_spin.value(),
            "random_state": self.random_state_spin.value(),
            "algorithm": self.algorithm_combo.currentText(),
            "normalize": self.normalize_check.isChecked(),
            "normalize_stack": self.normalize_stack_check.isChecked(),
            "tol": self.tol_spin.value()
        }

class GMMParameterDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.params = {
            "n_components": 8,
            "covariance_type": "full",
            "tol": 1e-3,
            "max_iter": 100,
            "random_state": 0,
            "normalize": False,
            "normalize_stack": False
        }
        self.setup_ui()

    def setup_ui(self):
        self.setWindowTitle("GMM Parameters")
        layout = QVBoxLayout(self)
        form_layout = QFormLayout()

        self.n_components_spin = QSpinBox()
        self.n_components_spin.setRange(1, 100)
        self.n_components_spin.setValue(self.params["n_components"])
        form_layout.addRow("Number of Components:", self.n_components_spin)

        self.covariance_combo = QComboBox()
        self.covariance_combo.addItems(["full", "tied", "diag", "spherical"])
        self.covariance_combo.setCurrentText(self.params["covariance_type"])
        form_layout.addRow("Covariance Type:", self.covariance_combo)

        self.tol_spin = QDoubleSpinBox()
        self.tol_spin.setRange(0, 1)
        self.tol_spin.setDecimals(6)
        self.tol_spin.setSingleStep(0.0001)
        self.tol_spin.setValue(self.params["tol"])
        form_layout.addRow("Tolerance:", self.tol_spin)

        self.max_iter_spin = QSpinBox()
        self.max_iter_spin.setRange(1, 10000)
        self.max_iter_spin.setValue(self.params["max_iter"])
        form_layout.addRow("Max Iterations:", self.max_iter_spin)

        self.random_state_spin = QSpinBox()
        self.random_state_spin.setRange(0, 1000000)
        self.random_state_spin.setValue(self.params["random_state"])
        form_layout.addRow("Random State:", self.random_state_spin)

        self.normalize_check = QCheckBox()
        self.normalize_check.setChecked(self.params["normalize"])
        form_layout.addRow("Normalize (per image):", self.normalize_check)

        self.normalize_stack_check = QCheckBox()
        self.normalize_stack_check.setChecked(self.params["normalize_stack"])
        form_layout.addRow("Normalize (entire stack):", self.normalize_stack_check)

        layout.addLayout(form_layout)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_params(self):
        return {
            "n_components": self.n_components_spin.value(),
            "covariance_type": self.covariance_combo.currentText(),
            "tol": self.tol_spin.value(),
            "max_iter": self.max_iter_spin.value(),
            "random_state": self.random_state_spin.value(),
            "normalize": self.normalize_check.isChecked(),
            "normalize_stack": self.normalize_stack_check.isChecked()
        }

class ThresholdParameterDialog(QDialog):
    params_changed = Signal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Threshold Parameters")
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        form_layout = QFormLayout()

        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0, 1000000)
        self.threshold_spin.setValue(128.0)
        self.threshold_spin.valueChanged.connect(self._emit_params)
        form_layout.addRow("Threshold:", self.threshold_spin)

        self.normalize_check = QCheckBox("Normalize stack before max projection")
        self.normalize_check.setChecked(False)
        self.normalize_check.toggled.connect(self._emit_params)
        form_layout.addRow(self.normalize_check)

        layout.addLayout(form_layout)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _emit_params(self):
        self.params_changed.emit(self.get_params())

    def get_params(self):
        return {
            "threshold": self.threshold_spin.value(),
            "normalize": self.normalize_check.isChecked()
        }

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CMP Viewer")
        self.resize(1200, 800)
        self.asset_manager = AssetManager()
        self.image_handler = ImageDisplayHandler()
        self.working_dir = None
        self.cached_composite = None
        self.visible_masks = set()
        self.mask_opacity = 0.5
        self.preview_mask = None
        self.preview_color = QColor(255, 255, 255) # White for preview
        self.graphs_window = None

        self._create_menu_bar()
        self._create_status_bar()
        self._setup_ui()

    def _setup_ui(self):
        self.mdi_area = QMdiArea()
        self.mdi_area.setBackground(QBrush(QColor(255, 255, 255)))
        self.setCentralWidget(self.mdi_area)

        dock = QDockWidget("Assets", self)
        dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.addDockWidget(Qt.LeftDockWidgetArea, dock)

        dock_container = QWidget()
        dock_layout = QVBoxLayout(dock_container)
        
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
        self.delete_images_btn = QPushButton("Delete")
        self.delete_images_btn.setFixedWidth(60)
        self.delete_images_btn.clicked.connect(self._delete_selected_images)
        
        image_btn_layout.addWidget(self.select_all_images_btn)
        image_btn_layout.addWidget(self.select_none_images_btn)
        image_btn_layout.addWidget(self.delete_images_btn)
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
        self.delete_masks_btn = QPushButton("Delete")
        self.delete_masks_btn.setFixedWidth(60)
        self.delete_masks_btn.clicked.connect(self._delete_selected_masks)
        
        mask_btn_layout.addWidget(self.select_all_masks_btn)
        mask_btn_layout.addWidget(self.select_none_masks_btn)
        mask_btn_layout.addWidget(self.delete_masks_btn)
        mask_btn_layout.addStretch()

        self.mask_list = QListWidget()
        self.mask_list.setSelectionMode(QListWidget.MultiSelection)
        self.mask_list.itemClicked.connect(self._mask_clicked)
        
        mask_h_layout.addLayout(mask_btn_layout)
        mask_h_layout.addWidget(self.mask_list)
        
        dock_layout.addWidget(QLabel("Cluster Masks:"))
        dock_layout.addWidget(mask_container)

        graph_container = QWidget()
        graph_h_layout = QHBoxLayout(graph_container)
        graph_h_layout.setContentsMargins(0, 0, 0, 0)
        
        graph_btn_layout = QVBoxLayout()
        graph_btn_layout.setContentsMargins(0, 0, 5, 0)
        
        self.select_all_graphs_btn = QPushButton("Select\nAll")
        self.select_all_graphs_btn.setFixedWidth(60)
        self.select_all_graphs_btn.clicked.connect(self._select_all_graphs)
        self.select_none_graphs_btn = QPushButton("Select\nNone")
        self.select_none_graphs_btn.setFixedWidth(60)
        self.select_none_graphs_btn.clicked.connect(self._select_none_graphs)
        self.delete_graphs_btn = QPushButton("Delete")
        self.delete_graphs_btn.setFixedWidth(60)
        self.delete_graphs_btn.clicked.connect(self._delete_selected_graphs)
        self.save_graphs_btn = QPushButton("Save")
        self.save_graphs_btn.setFixedWidth(60)
        self.save_graphs_btn.clicked.connect(self._save_selected_graphs)
        
        graph_btn_layout.addWidget(self.select_all_graphs_btn)
        graph_btn_layout.addWidget(self.select_none_graphs_btn)
        graph_btn_layout.addWidget(self.delete_graphs_btn)
        graph_btn_layout.addWidget(self.save_graphs_btn)
        graph_btn_layout.addStretch()

        self.graph_list = QListWidget()
        self.graph_list.setSelectionMode(QListWidget.MultiSelection)
        self.graph_list.itemClicked.connect(self._graph_clicked)
        
        graph_h_layout.addLayout(graph_btn_layout)
        graph_h_layout.addWidget(self.graph_list)
        
        dock_layout.addWidget(QLabel("Graphs:"))
        dock_layout.addWidget(graph_container)

        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setValue(50)
        self.opacity_slider.valueChanged.connect(self._opacity_changed)
        dock_layout.addWidget(QLabel("Mask Opacity:"))
        dock_layout.addWidget(self.opacity_slider)

        self.save_button = QPushButton("Save Visible")
        self.save_button.clicked.connect(self._save_visible)
        dock_layout.addWidget(self.save_button)

        dock.setWidget(dock_container)

        self.viewer_subwindow = QMdiSubWindow()
        self.viewer_subwindow.setAttribute(Qt.WA_DeleteOnClose, False)
        self.viewer_subwindow.setWindowTitle("Image Viewer")
        self.viewer_view = ZoomableView()
        self.viewer_subwindow.setWidget(self.viewer_view)
        self.mdi_area.addSubWindow(self.viewer_subwindow)
        self.viewer_subwindow.show()

        self.bg_label = QLabel(
            "1.  Click 'Home' and navigate to your image folder.  \n"
            "2.  Select images in left-column to appear in Image window.  Right-click to access submenu to change image color or contrast.  \n"
            "3.  Use ctrl+scroll to zoom in and out.  Click and drag to create a crop area.  When crop selection is placed, use Tools>Crop to apply crop to all images.",
            self.mdi_area.viewport()
        )
        self.bg_label.setWordWrap(True)
        self.bg_label.setAlignment(Qt.AlignCenter)
        self.bg_label.setStyleSheet("QLabel { color: rgba(0, 0, 0, 255); font-size: 18px; padding: 20px; }")
        self.bg_label.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.bg_label.show()
        self.bg_label.lower()
        self._center_bg_label()

    def _center_bg_label(self):
        if hasattr(self, 'bg_label') and hasattr(self, 'mdi_area'):
            self.bg_label.lower()
            w = int(self.mdi_area.viewport().width() * 0.8)
            h = int(self.mdi_area.viewport().height() * 0.8)
            self.bg_label.setFixedSize(w, h)
            self.bg_label.move((self.mdi_area.viewport().width() - w) // 2, (self.mdi_area.viewport().height() - h) // 2)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._center_bg_label()

    def _create_menu_bar(self):
        menu_bar = self.menuBar()
        home_action = QAction("Home", self)
        home_action.triggered.connect(self._home_triggered)
        menu_bar.addAction(home_action)

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

        filters_menu = tools_menu.addMenu("Filters")
        for f in ["gaussian", "median", "mean", "blur", "unsharp"]:
            action = QAction(f.capitalize(), self)
            action.triggered.connect(lambda checked=False, name=f: self._apply_filter_to_visible(name))
            filters_menu.addAction(action)

        tools_menu.addSeparator()
        export_images_action = QAction("Export Modified Images", self)
        export_images_action.triggered.connect(self._export_modified_images)
        tools_menu.addAction(export_images_action)

        stack_action = QAction("Stack Images", self)
        stack_action.triggered.connect(self._stack_images)
        tools_menu.addAction(stack_action)

        threshold_mask_action = QAction("Mask From Threshold", self)
        threshold_mask_action.triggered.connect(self._create_threshold_mask)
        tools_menu.addAction(threshold_mask_action)

        cluster_menu = menu_bar.addMenu("Cluster")
        kmeans_action = QAction("k-means", self)
        kmeans_action.triggered.connect(self._run_kmeans)
        cluster_menu.addAction(kmeans_action)

        gmm_action = QAction("Gaussian Mixture", self)
        gmm_action.triggered.connect(self._run_gmm)
        cluster_menu.addAction(gmm_action)

        analyze_menu = menu_bar.addMenu("Analyze")
        create_hist_action = QAction("Create Histograms", self)
        create_hist_action.triggered.connect(self._create_histograms_from_selection)
        analyze_menu.addAction(create_hist_action)

    def _home_triggered(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Working Directory")
        if directory:
            self.working_dir = directory
            self.asset_manager.set_working_dir(directory)
            self.image_handler.clear()
            self.visible_masks.clear()
            self._update_asset_list()
            self._update_mask_list()
            self._update_graph_list()
            self._refresh_viewer()
            self.bg_label.lower()

    def _update_asset_list(self):
        # Block signals to prevent _asset_clicked from being triggered during refresh
        self.image_list.blockSignals(True)
        self.image_list.clear()
        for name in self.asset_manager.get_image_list():
            asset = self.asset_manager.get_image_by_name(name)
            item = QListWidgetItem(name)
            qimg = asset.to_qimage()
            item.setIcon(QPixmap.fromImage(qimg.scaled(100, 100, Qt.KeepAspectRatio)))
            self.image_list.addItem(item)
            if self.image_handler.is_visible(name):
                item.setSelected(True)
        self.image_list.blockSignals(False)

    def _update_mask_list(self):
        self.mask_list.blockSignals(True)
        self.mask_list.clear()
        if not self.working_dir:
            self.mask_list.blockSignals(False)
            return
        mask_dir = os.path.join(self.working_dir, "Cluster Masks")
        if os.path.exists(mask_dir):
            masks = [f for f in os.listdir(mask_dir) if (f.startswith("KC_") or f.startswith("GMM_") or f.startswith("ThresholdMask_")) and f.endswith(".npy")]
            # Sort masks
            try:
                masks.sort()
            except:
                pass
            for mask_name in masks:
                item = QListWidgetItem(mask_name)
                self.mask_list.addItem(item)
                if mask_name in self.visible_masks:
                    item.setSelected(True)
        self.mask_list.blockSignals(False)

    def _update_graph_list(self):
        self.graph_list.blockSignals(True)
        self.graph_list.clear()
        if not self.working_dir:
            self.graph_list.blockSignals(False)
            return
        graph_dir = os.path.join(self.working_dir, "Graphs")
        if os.path.exists(graph_dir):
            graphs = [f for f in os.listdir(graph_dir) if (f.startswith("Hist_") or f.startswith("Hist_Overlay_")) and f.endswith(".png")]
            try:
                graphs.sort()
            except:
                pass
            for graph_name in graphs:
                item = QListWidgetItem(graph_name)
                self.graph_list.addItem(item)
        self.graph_list.blockSignals(False)

    def _create_histograms_from_selection(self):
        if not self.working_dir:
            QMessageBox.warning(self, "Graphing", "Please select a working directory first.")
            return

        selected_masks = self.mask_list.selectedItems()
        if len(selected_masks) != 1:
            QMessageBox.warning(self, "Graphing", "Please select exactly one cluster mask.")
            return
        
        mask_name = selected_masks[0].text()
        mask_path = os.path.join(self.working_dir, "Cluster Masks", mask_name)

        selected_images = self.image_list.selectedItems()
        if not selected_images:
            image_names = list(self.image_handler.visible_assets)
        else:
            image_names = [item.text() for item in selected_images]

        if not image_names:
            QMessageBox.warning(self, "Graphing", "No images selected or visible.")
            return

        try:
            self.statusBar().showMessage("Calculating measurements and generating histograms...")
            QApplication.setOverrideCursor(Qt.WaitCursor)
            
            measurements = graphing.calculate_mask_measurements(self.asset_manager, image_names, mask_path)
            if not measurements:
                QMessageBox.critical(self, "Graphing", "Failed to calculate measurements.")
                return

            graph_dir = os.path.join(self.working_dir, "Graphs")
            graphing.create_histograms(measurements, mask_name, graph_dir)
            graphing.create_overlaid_histogram(measurements, mask_name, graph_dir)
            graphing.save_measurements_json(measurements, mask_name, graph_dir)

            self.statusBar().showMessage("Graphing completed.", 3000)
            self._update_graph_list()
            self._show_graphs_window()
        except Exception as e:
            QMessageBox.critical(self, "Graphing Error", f"An error occurred: {str(e)}")
        finally:
            QApplication.restoreOverrideCursor()

    def _show_graphs_window(self):
        if self.graphs_window is None or not self.graphs_window.isVisible():
            self.graphs_window = QMdiSubWindow()
            self.graphs_window.setWindowTitle("Graphs")
            self.graphs_window.resize(800, 600)
            
            scroll_area = QWidget()
            self.graphs_layout = QVBoxLayout(scroll_area)
            
            from PySide6.QtWidgets import QScrollArea
            sa = QScrollArea()
            sa.setWidgetResizable(True)
            sa.setWidget(scroll_area)
            
            self.graphs_window.setWidget(sa)
            self.mdi_area.addSubWindow(self.graphs_window)
            self.graphs_window.show()
        
        # Clear existing graphs in the window
        for i in reversed(range(self.graphs_layout.count())): 
            self.graphs_layout.itemAt(i).widget().setParent(None)

        selected_graphs = self.graph_list.selectedItems()
        if not selected_graphs:
            return
        
        graph_dir = os.path.join(self.working_dir, "Graphs")
        
        if len(selected_graphs) == 1:
            # Single graph - show the pre-rendered image
            graph_path = os.path.join(graph_dir, selected_graphs[0].text())
            if os.path.exists(graph_path):
                label = QLabel()
                pixmap = QPixmap(graph_path)
                label.setPixmap(pixmap)
                self.graphs_layout.addWidget(label)
        else:
            # Multi-selection - create a combined histogram
            items_measurements = []
            for item in selected_graphs:
                name = item.text()
                # Try to parse the name: Hist_<image>_<mask_no_ext>.png
                # Or Hist_Overlay_<mask_no_ext>.png
                if name.startswith("Hist_Overlay_"):
                    # For overlays, we'd need to load all images for that mask.
                    # But the user might want specific individual graphs.
                    # Let's see if we can extract measurements for each selected item.
                    pass
                elif name.startswith("Hist_"):
                    # Pattern: Hist_ImageName_MaskName.png
                    # This is tricky because image name and mask name can contain underscores.
                    # Let's try to find matching JSON files and extract data.
                    # Actually, a better way is to iterate through all JSONs in Graphs dir.
                    
                    # For now, let's assume we can find the image name and mask name.
                    # A more robust way:
                    found = False
                    for f in os.listdir(graph_dir):
                        if f.startswith("Measurements_") and f.endswith(".json"):
                            mask_part = f[len("Measurements_"):-len(".json")]
                            if mask_part in name:
                                with open(os.path.join(graph_dir, f), 'r') as jf:
                                    measurements = json.load(jf)
                                    # Find which image matches this name
                                    for img_name, values in measurements.items():
                                        safe_img = "".join([c if c.isalnum() or c in (' ', '.', '_', '-') else '_' for c in img_name])
                                        if safe_img in name:
                                            items_measurements.append((img_name, values))
                                            found = True
                                            break
                            if found: break
            
            if items_measurements:
                combined_rgb = graphing.create_dynamic_overlaid_histogram(items_measurements)
                if combined_rgb is not None:
                    # Ensure the array is C-contiguous for QImage
                    combined_rgb = np.ascontiguousarray(combined_rgb)
                    h, w, _ = combined_rgb.shape
                    qimg = QImage(combined_rgb.data, w, h, combined_rgb.strides[0], QImage.Format_RGB888)
                    label = QLabel()
                    label.setPixmap(QPixmap.fromImage(qimg))
                    self.graphs_layout.addWidget(label)

    def _save_selected_graphs(self):
        if not self.working_dir: return
        selected = self.graph_list.selectedItems()
        if not selected:
            QMessageBox.warning(self, "Save", "No graphs selected.")
            return

        path, _ = QFileDialog.getSaveFileName(self, "Save Combined Histogram", "", "PNG (*.png)")
        if not path: return

        graph_dir = os.path.join(self.working_dir, "Graphs")
        if len(selected) == 1:
            # Just copy the file
            import shutil
            shutil.copy(os.path.join(graph_dir, selected[0].text()), path)
        else:
            # Create and save combined
            items_measurements = []
            for item in selected:
                name = item.text()
                for f in os.listdir(graph_dir):
                    if f.startswith("Measurements_") and f.endswith(".json"):
                        mask_part = f[len("Measurements_"):-len(".json")]
                        if mask_part in name:
                            with open(os.path.join(graph_dir, f), 'r') as jf:
                                measurements = json.load(jf)
                                for img_name, values in measurements.items():
                                    safe_img = "".join([c if c.isalnum() or c in (' ', '.', '_', '-') else '_' for c in img_name])
                                    if safe_img in name:
                                        items_measurements.append((img_name, values))
                                        break
                
            if items_measurements:
                graphing.create_dynamic_overlaid_histogram(items_measurements, output_path=path)
                QMessageBox.information(self, "Save", f"Combined histogram saved to {path}")

    def _graph_clicked(self, item):
        self._show_graphs_window()

    def _select_all_graphs(self):
        for i in range(self.graph_list.count()):
            self.graph_list.item(i).setSelected(True)
        self._show_graphs_window()

    def _select_none_graphs(self):
        self.graph_list.clearSelection()
        self._show_graphs_window()

    def _delete_selected_graphs(self):
        selected = self.graph_list.selectedItems()
        if not selected: return
        if QMessageBox.question(self, "Delete", f"Delete {len(selected)} graphs?", QMessageBox.Yes|QMessageBox.No) == QMessageBox.Yes:
            graph_dir = os.path.join(self.working_dir, "Graphs")
            for item in selected:
                name = item.text()
                path = os.path.join(graph_dir, name)
                if os.path.exists(path):
                    os.remove(path)
            self._update_graph_list()
            self._show_graphs_window()

    def _run_kmeans(self):
        if not self.working_dir:
            QMessageBox.warning(self, "K-Means", "Please select a working directory first.")
            return

        selected_items = self.image_list.selectedItems()
        if not selected_items:
            image_names = list(self.image_handler.visible_assets)
        else:
            image_names = [item.text() for item in selected_items]

        if not image_names:
            QMessageBox.warning(self, "K-Means", "No images selected or visible to cluster.")
            return

        stack = image_stacker.load_and_stack_images(self.asset_manager, image_names)
        if stack is None:
            QMessageBox.critical(self, "K-Means", "Failed to create image stack.")
            return

        # Ensure stack is float32
        stack = stack.astype(np.float32)

        dialog = ClusterParameterDialog(self)
        if dialog.exec() == QDialog.Accepted:
            params = dialog.get_params()
            try:
                self.statusBar().showMessage("Running K-Means...")
                QApplication.setOverrideCursor(Qt.WaitCursor)
                cluster_mask = clustering.kmeans_clustering(stack, **params)
                individual_masks = clustering.get_individual_masks(cluster_mask, params["n_clusters"])

                mask_dir = os.path.join(self.working_dir, "Cluster Masks")
                os.makedirs(mask_dir, exist_ok=True)

                for i, mask in enumerate(individual_masks):
                    mask_name = f"KC_{i+1}.npy"
                    np.save(os.path.join(mask_dir, mask_name), mask)

                self.statusBar().showMessage("K-Means completed.", 3000)
                self._update_mask_list()
            except Exception as e:
                QMessageBox.critical(self, "K-Means Error", f"An error occurred during clustering: {str(e)}")
            finally:
                QApplication.restoreOverrideCursor()

    def _run_gmm(self):
        if not self.working_dir:
            QMessageBox.warning(self, "Gaussian Mixture", "Please select a working directory first.")
            return

        selected_items = self.image_list.selectedItems()
        if not selected_items:
            image_names = list(self.image_handler.visible_assets)
        else:
            image_names = [item.text() for item in selected_items]

        if not image_names:
            QMessageBox.warning(self, "Gaussian Mixture", "No images selected or visible to cluster.")
            return

        stack = image_stacker.load_and_stack_images(self.asset_manager, image_names)
        if stack is None:
            QMessageBox.critical(self, "Gaussian Mixture", "Failed to create image stack.")
            return

        # Ensure stack is float32
        stack = stack.astype(np.float32)

        dialog = GMMParameterDialog(self)
        if dialog.exec() == QDialog.Accepted:
            params = dialog.get_params()
            try:
                self.statusBar().showMessage("Running Gaussian Mixture...")
                QApplication.setOverrideCursor(Qt.WaitCursor)
                cluster_mask = clustering.gaussian_mixture_clustering(stack, **params)
                individual_masks = clustering.get_individual_masks(cluster_mask, params["n_components"])

                mask_dir = os.path.join(self.working_dir, "Cluster Masks")
                os.makedirs(mask_dir, exist_ok=True)

                for i, mask in enumerate(individual_masks):
                    mask_name = f"GMM_{i+1}.npy"
                    np.save(os.path.join(mask_dir, mask_name), mask)

                self.statusBar().showMessage("Gaussian Mixture completed.", 3000)
                self._update_mask_list()
            except Exception as e:
                QMessageBox.critical(self, "GMM Error", f"An error occurred during clustering: {str(e)}")
            finally:
                QApplication.restoreOverrideCursor()

    def _create_threshold_mask(self):
        if not self.working_dir:
            QMessageBox.warning(self, "Threshold Mask", "Please select a working directory first.")
            return

        selected_items = self.image_list.selectedItems()
        if not selected_items:
            image_names = list(self.image_handler.visible_assets)
        else:
            image_names = [item.text() for item in selected_items]

        if not image_names:
            QMessageBox.warning(self, "Threshold Mask", "No images selected or visible.")
            return

        stack = image_stacker.load_and_stack_images(self.asset_manager, image_names)
        if stack is None:
            QMessageBox.critical(self, "Threshold Mask", "Failed to create image stack.")
            return

        # Ensure stack is float32
        stack = stack.astype(np.float32)

        dialog = ThresholdParameterDialog(self)
        
        def update_preview(params):
            temp_stack = stack
            if params["normalize"]:
                s_min, s_max = temp_stack.min(), temp_stack.max()
                if s_max > s_min:
                    temp_stack = (temp_stack - s_min) / (s_max - s_min)
                else:
                    temp_stack = np.zeros_like(temp_stack)
            
            max_proj = np.max(temp_stack, axis=0)
            self.preview_mask = (max_proj > params["threshold"]).astype(np.uint8)
            self.cached_composite = None
            self._refresh_viewer()

        dialog.params_changed.connect(update_preview)
        # Show initial preview
        update_preview(dialog.get_params())

        if dialog.exec() == QDialog.Accepted:
            params = dialog.get_params()
            try:
                self.statusBar().showMessage("Creating threshold mask...")
                QApplication.setOverrideCursor(Qt.WaitCursor)
                
                # Use the preview mask if it was just generated, or regenerate to be sure
                if params["normalize"]:
                    s_min, s_max = stack.min(), stack.max()
                    if s_max > s_min:
                        stack = (stack - s_min) / (s_max - s_min)
                    else:
                        stack = np.zeros_like(stack)
                
                max_projection = np.max(stack, axis=0)
                binary_mask = (max_projection > params["threshold"]).astype(np.uint8)

                mask_dir = os.path.join(self.working_dir, "Cluster Masks")
                os.makedirs(mask_dir, exist_ok=True)

                # Find a unique name for the threshold mask
                idx = 1
                while os.path.exists(os.path.join(mask_dir, f"ThresholdMask_{idx}.npy")):
                    idx += 1
                mask_name = f"ThresholdMask_{idx}.npy"
                np.save(os.path.join(mask_dir, mask_name), binary_mask)

                self.statusBar().showMessage(f"Threshold mask {mask_name} created.", 3000)
                self._update_mask_list()
            except Exception as e:
                QMessageBox.critical(self, "Threshold Error", f"An error occurred: {str(e)}")
            finally:
                QApplication.restoreOverrideCursor()
        
        self.preview_mask = None
        self.cached_composite = None
        self._refresh_viewer()

    def _mask_clicked(self, item):
        mask_name = item.text()
        if mask_name in self.visible_masks:
            self.visible_masks.remove(mask_name)
        else:
            self.visible_masks.add(mask_name)
        self.cached_composite = None
        self._refresh_viewer()

    def _opacity_changed(self, value):
        self.mask_opacity = value / 100.0
        self.cached_composite = None
        self._refresh_viewer()

    def _show_image_context_menu(self, position):
        item = self.image_list.itemAt(position)
        if not item: return
        self._show_context_menu(self.image_list, position, False)

    def _show_context_menu(self, list_widget, position, is_mask):
        item = list_widget.itemAt(position)
        if not item: return
        name = item.text()
        asset = self.asset_manager.get_image_by_name(name)
        
        menu = QMenu()
        color_menu = menu.addMenu("Color")
        for color in ["grayscale", "red", "green", "blue", "cyan", "magenta", "yellow"]:
            action = QAction(color.capitalize(), self)
            action.triggered.connect(lambda checked=False, c=color: self._change_color(name, c, is_mask))
            color_menu.addAction(action)

        contrast_action = QAction("Contrast Stretch", self)
        contrast_action.setCheckable(True)
        contrast_action.setChecked(asset.pipeline.config.get("contrast_stretch", False))
        contrast_action.triggered.connect(lambda: self._toggle_transform(asset, "contrast_stretch"))
        menu.addAction(contrast_action)

        normalize_action = QAction("Normalize", self)
        normalize_action.setCheckable(True)
        normalize_action.setChecked(asset.pipeline.config.get("normalize", False))
        normalize_action.triggered.connect(lambda: self._toggle_transform(asset, "normalize"))
        menu.addAction(normalize_action)

        menu.exec(list_widget.mapToGlobal(position))

    def _change_color(self, name, color_name, is_mask=False):
        asset = self.asset_manager.get_image_by_name(name)
        if asset:
            asset.pipeline.config["color"] = color_name
            asset.save_project()
            self.cached_composite = None
            self._update_asset_list()
            self._refresh_viewer()

    def _toggle_transform(self, asset, key):
        asset.pipeline.config[key] = not asset.pipeline.config.get(key, False)
        asset.save_project()
        self.cached_composite = None
        self._update_asset_list()
        self._refresh_viewer()

    def _invert_selected_images(self):
        for item in self.image_list.selectedItems():
            asset = self.asset_manager.get_image_by_name(item.text())
            asset.pipeline.config["invert"] = not asset.pipeline.config.get("invert", False)
            asset.save_project()
        self.cached_composite = None
        self._update_asset_list()
        self._refresh_viewer()

    def _apply_filter_to_visible(self, filter_name):
        selected = self.image_list.selectedItems()
        if not selected: return
        
        first_asset = self.asset_manager.get_image_by_name(selected[0].text())
        initial_params = first_asset.pipeline.config.get("filter_params", {}).get(filter_name, {})
        
        dialog = FilterParameterDialog(filter_name, initial_params, self)
        if dialog.exec() == QDialog.Accepted:
            params = dialog.get_params()
            for item in selected:
                asset = self.asset_manager.get_image_by_name(item.text())
                if filter_name not in asset.pipeline.config["filters"]:
                    asset.pipeline.config["filters"].append(filter_name)
                if "filter_params" not in asset.pipeline.config:
                    asset.pipeline.config["filter_params"] = {}
                asset.pipeline.config["filter_params"][filter_name] = params
                asset.save_project()
            self.cached_composite = None
            self._update_asset_list()
            self._refresh_viewer()

    def _crop_all(self):
        rect = self.viewer_view.get_selection_rect()
        if not rect:
            QMessageBox.warning(self, "No Selection", "Please create a selection rectangle first.")
            return
        
        for name, asset in self.asset_manager.images.items():
            asset.pipeline.config.setdefault("transforms", []).append({"type": "crop", "params": rect})
            asset.save_project()
        
        self.cached_composite = None
        self._update_asset_list()
        self._refresh_viewer()
        self.viewer_view.clear_selection()

    def _rotate_all(self):
        angle, ok = QInputDialog.getDouble(self, "Rotate All", "Angle (degrees):", 0, -360, 360, 1)
        if ok:
            for name, asset in self.asset_manager.images.items():
                asset.pipeline.config.setdefault("transforms", []).append({"type": "rotate", "angle": angle})
                asset.save_project()
            self.cached_composite = None
            self._update_asset_list()
            self._refresh_viewer()

    def _export_modified_images(self):
        if not self.working_dir: return
        out_dir = QFileDialog.getExistingDirectory(self, "Select Export Directory")
        if not out_dir: return
        
        for name, asset in self.asset_manager.images.items():
            data = asset.get_rendered_data()
            if data.max() <= 1.01: data = (data * 255).astype(np.uint8)
            else: data = data.astype(np.uint8)
            Image.fromarray(data).save(os.path.join(out_dir, name))
        QMessageBox.information(self, "Export", "Images exported successfully.")

    def _stack_images(self):
        if not self.working_dir: return
        
        selected_items = self.image_list.selectedItems()
        if not selected_items:
            # If nothing selected, use all visible
            image_names = list(self.image_handler.visible_assets)
        else:
            image_names = [item.text() for item in selected_items]
            
        if not image_names:
            QMessageBox.warning(self, "Stacking", "No images selected or visible to stack.")
            return

        stack = image_stacker.load_and_stack_images(self.asset_manager, image_names)
        
        if stack is not None:
            save_path, _ = QFileDialog.getSaveFileName(self, "Save Stack", self.working_dir, "Numpy Stack (*.npy)")
            if save_path:
                image_stacker.save_stack(stack, save_path)
                QMessageBox.information(self, "Stacking", f"Successfully created stack with shape: {stack.shape} and saved to {save_path}")
        else:
            QMessageBox.critical(self, "Stacking", "Failed to create stack. Check console for details (possible dimension mismatch).")

    def _save_visible(self):
        if not self.working_dir: return
        path, _ = QFileDialog.getSaveFileName(self, "Save Composite", "", "PNG (*.png);;TIFF (*.tif)")
        if path:
            pixmap = self.viewer_view.pixmap_item.pixmap()
            if not pixmap.isNull():
                pixmap.save(path)

    def _select_all_images(self):
        for i in range(self.image_list.count()):
            item = self.image_list.item(i)
            if not self.image_handler.is_visible(item.text()):
                self.image_handler.toggle_visibility(item.text())
            item.setSelected(True)
        self.cached_composite = None
        self._refresh_viewer()

    def _select_none_images(self):
        self.image_list.clearSelection()
        self.image_handler.clear()
        self.cached_composite = None
        self._refresh_viewer()

    def _delete_selected_images(self):
        selected = self.image_list.selectedItems()
        if not selected: return
        if QMessageBox.question(self, "Delete", f"Delete {len(selected)} images?", QMessageBox.Yes|QMessageBox.No) == QMessageBox.Yes:
            for item in selected:
                name = item.text()
                self.asset_manager.delete_image(name)
                self.image_handler.remove_asset(name)
            self._update_asset_list()
            self.cached_composite = None
            self._refresh_viewer()

    def _select_all_masks(self):
        for i in range(self.mask_list.count()):
            item = self.mask_list.item(i)
            self.visible_masks.add(item.text())
            item.setSelected(True)
        self.cached_composite = None
        self._refresh_viewer()

    def _select_none_masks(self):
        self.mask_list.clearSelection()
        self.visible_masks.clear()
        self.cached_composite = None
        self._refresh_viewer()

    def _delete_selected_masks(self):
        selected = self.mask_list.selectedItems()
        if not selected: return
        if QMessageBox.question(self, "Delete", f"Delete {len(selected)} cluster masks?", QMessageBox.Yes|QMessageBox.No) == QMessageBox.Yes:
            mask_dir = os.path.join(self.working_dir, "Cluster Masks")
            for item in selected:
                name = item.text()
                path = os.path.join(mask_dir, name)
                if os.path.exists(path):
                    os.remove(path)
                if name in self.visible_masks:
                    self.visible_masks.remove(name)
            self._update_mask_list()
            self.cached_composite = None
            self._refresh_viewer()

    def _asset_clicked(self, item):
        name = item.text()
        self.image_handler.toggle_visibility(name)
        self.cached_composite = None
        self._refresh_viewer()

    def _refresh_viewer(self):
        visible_names = self.image_handler.visible_assets
        if not visible_names and not self.visible_masks and self.preview_mask is None:
            self.viewer_view.set_pixmap(None)
            self.bg_label.lower()
            return

        if self.cached_composite is None:
            self.cached_composite = self.image_handler.render_composite(self.asset_manager)
            
            # Overlay masks
            if (self.visible_masks and self.working_dir) or self.preview_mask is not None:
                if self.cached_composite is None:
                    # If no images, we need a blank slate based on mask size
                    # But we don't know the size yet. Let's load first mask to see.
                    if self.preview_mask is not None:
                        h, w = self.preview_mask.shape
                    else:
                        first_mask_path = os.path.join(self.working_dir, "Cluster Masks", list(self.visible_masks)[0])
                        if os.path.exists(first_mask_path):
                            m = np.load(first_mask_path)
                            h, w = m.shape
                        else:
                            return
                    composite_rgba = np.zeros((h, w, 4), dtype=np.float32)
                else:
                    # Convert QImage to numpy RGBA
                    qimg = self.cached_composite.convertToFormat(QImage.Format_RGBA8888)
                    width = qimg.width()
                    height = qimg.height()
                    ptr = qimg.constBits()
                    # QImage format RGBA8888 is 4 bytes per pixel
                    arr = np.array(ptr).reshape(height, width, 4).astype(np.float32) / 255.0
                    composite_rgba = arr

                if self.visible_masks and self.working_dir:
                    for mask_name in sorted(self.visible_masks):
                        mask_path = os.path.join(self.working_dir, "Cluster Masks", mask_name)
                        if os.path.exists(mask_path):
                            mask = np.load(mask_path)
                            # We need to make sure mask size matches composite size
                            if mask.shape != composite_rgba.shape[:2]:
                                # Optionally resize mask here? For now assume matching.
                                continue
                            
                            # Generate a color for the mask
                            try:
                                if mask_name.startswith("KC_"):
                                    idx = int(mask_name.split('_')[1].split('.')[0])
                                elif mask_name.startswith("ThresholdMask_"):
                                    idx = int(mask_name.split('_')[1].split('.')[0]) + 100 # Offset to avoid same colors
                                else:
                                    idx = hash(mask_name)
                            except:
                                idx = hash(mask_name)
                            color_hue = (idx * 137.5) % 360 # Golden angle for distinct colors
                            color = QColor.fromHsvF(color_hue/360.0, 1.0, 1.0)
                            r, g, b = color.redF(), color.greenF(), color.blueF()
                            
                            # Blend mask
                            mask_bool = mask.astype(bool)
                            composite_rgba[mask_bool, 0] = composite_rgba[mask_bool, 0] * (1 - self.mask_opacity) + r * self.mask_opacity
                            composite_rgba[mask_bool, 1] = composite_rgba[mask_bool, 1] * (1 - self.mask_opacity) + g * self.mask_opacity
                            composite_rgba[mask_bool, 2] = composite_rgba[mask_bool, 2] * (1 - self.mask_opacity) + b * self.mask_opacity

                if self.preview_mask is not None:
                    mask = self.preview_mask
                    if mask.shape == composite_rgba.shape[:2]:
                        r, g, b = self.preview_color.redF(), self.preview_color.greenF(), self.preview_color.blueF()
                        mask_bool = mask.astype(bool)
                        # We can use a different opacity for preview or the same. 
                        # Let's use 0.7 for higher visibility during adjustment
                        preview_opacity = 0.7
                        composite_rgba[mask_bool, 0] = composite_rgba[mask_bool, 0] * (1 - preview_opacity) + r * preview_opacity
                        composite_rgba[mask_bool, 1] = composite_rgba[mask_bool, 1] * (1 - preview_opacity) + g * preview_opacity
                        composite_rgba[mask_bool, 2] = composite_rgba[mask_bool, 2] * (1 - preview_opacity) + b * preview_opacity
                
                # Convert back to QImage
                display_img = (composite_rgba * 255).astype(np.uint8)
                h, w, _ = display_img.shape
                self.cached_composite = QImage(display_img.data, w, h, display_img.strides[0], QImage.Format_RGBA8888).copy()

        self.viewer_view.set_pixmap(self.cached_composite)
        self.bg_label.lower()

    def _create_status_bar(self):
        self.statusBar().showMessage("Ready")
