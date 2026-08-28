from PySide6.QtWidgets import (
    QMainWindow, QMenu, QMenuBar, QFileDialog, QListWidget, QListWidgetItem, 
    QWidget, QVBoxLayout, QLabel, QMdiArea, QMdiSubWindow, QDockWidget,
    QMessageBox, QPushButton, QHBoxLayout, QFormLayout, QDoubleSpinBox, 
    QSpinBox, QDialogButtonBox, QLineEdit, QComboBox, QInputDialog, QSlider,
    QApplication, QCheckBox, QDialog
)
from PySide6.QtGui import QAction, QPixmap, QPainter, QPalette, QPen, QColor, QBrush, QImage
from PySide6.QtCore import Qt, QSize, QPoint, QPointF, QRectF, Signal, QObject, QThread, QSettings
import os
import numpy as np
import pandas as pd
import cv2
import json
import scipy.stats
import qimage2ndarray
from assets import AssetManager
from image_handler import ImageDisplayHandler
import image_stacker
import clustering
import measure_utilities
import histogram_plots
import kde_plots
import export_plot_utils
import mask_refinement
from reconciliation import reconcile_project_with_disk
from workers import ClusteringWorker
from widgets import ZoomableView
from dialogs import (
    FilterParameterDialog, ClusterParameterDialog, ISODATAParameterDialog,
    GMMParameterDialog, ThresholdParameterDialog, JointPlotDialog,
    RefineMaskDialog, MaskPropertiesDialog
)

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
        self.stop_btn = None

        self._create_menu_bar()
        self._create_status_bar()
        self._setup_ui()

        # Load last working directory at startup
        QSettings.setDefaultFormat(QSettings.IniFormat)
        settings = QSettings("CMP", "Viewer")
        last_dir = settings.value("last_working_dir")
        if last_dir and os.path.exists(last_dir):
            # Use a single-shot timer to allow the UI to show before the dialog potentially pops up
            from PySide6.QtCore import QTimer
            QTimer.singleShot(0, lambda: self._load_working_directory(last_dir))

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
        self.save_masks_btn = QPushButton("Save")
        self.save_masks_btn.setFixedWidth(60)
        self.save_masks_btn.clicked.connect(self._save_visible_masks)
        self.merge_masks_btn = QPushButton("Merge\nMasks")
        self.merge_masks_btn.setFixedWidth(60)
        self.merge_masks_btn.clicked.connect(self._merge_selected_masks)
        
        mask_btn_layout.addWidget(self.select_all_masks_btn)
        mask_btn_layout.addWidget(self.select_none_masks_btn)
        mask_btn_layout.addWidget(self.delete_masks_btn)
        mask_btn_layout.addWidget(self.save_masks_btn)
        mask_btn_layout.addWidget(self.merge_masks_btn)
        mask_btn_layout.addStretch()

        self.mask_list = QListWidget()
        self.mask_list.setIconSize(QSize(100, 50))
        self.mask_list.setSelectionMode(QListWidget.MultiSelection)
        self.mask_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.mask_list.customContextMenuRequested.connect(self._show_mask_context_menu)
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
        self.save_graphs_btn = QPushButton("Save\nPNG")
        self.save_graphs_btn.setFixedWidth(60)
        self.save_graphs_btn.clicked.connect(self._save_selected_graphs)
        self.export_graphs_btn = QPushButton("Export\nCSV")
        self.export_graphs_btn.setFixedWidth(60)
        self.export_graphs_btn.clicked.connect(self._export_selected_graphs)
        
        graph_btn_layout.addWidget(self.select_all_graphs_btn)
        graph_btn_layout.addWidget(self.select_none_graphs_btn)
        graph_btn_layout.addWidget(self.delete_graphs_btn)
        graph_btn_layout.addWidget(self.save_graphs_btn)
        graph_btn_layout.addWidget(self.export_graphs_btn)
        graph_btn_layout.addStretch()

        self.graph_list = QListWidget()
        self.graph_list.setSelectionMode(QListWidget.MultiSelection)
        self.graph_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.graph_list.customContextMenuRequested.connect(self._show_graph_context_menu)
        self.graph_list.itemClicked.connect(self._graph_clicked)
        self.graph_list.itemSelectionChanged.connect(self._show_graphs_window)
        
        graph_h_layout.addLayout(graph_btn_layout)
        graph_h_layout.addWidget(self.graph_list)
        
        dock_layout.addWidget(QLabel("Graphs:"))
        dock_layout.addWidget(graph_container)

        self.kde_checkbox = QCheckBox("Show KDE / High Quality")
        self.kde_checkbox.stateChanged.connect(self._show_graphs_window)
        dock_layout.addWidget(self.kde_checkbox)

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

        file_menu = menu_bar.addMenu("File")
        import_image_action = QAction("Import Image", self)
        import_image_action.triggered.connect(self._import_image)
        file_menu.addAction(import_image_action)

        import_mask_action = QAction("Import Mask", self)
        import_mask_action.triggered.connect(self._import_mask)
        file_menu.addAction(import_mask_action)

        tools_menu = menu_bar.addMenu("Tools")
        
        invert_action = QAction("Invert All", self)
        invert_action.triggered.connect(self._invert_all_images)
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
            action.triggered.connect(lambda checked=False, name=f: self._apply_filter_to_all(name))
            filters_menu.addAction(action)

        image_adjustments_menu = tools_menu.addMenu("Image Adjustments")
        undo_invert_action = QAction("Undo Invert All", self)
        undo_invert_action.triggered.connect(self._undo_invert_all)
        image_adjustments_menu.addAction(undo_invert_action)

        undo_rotation_action = QAction("Undo Rotation All", self)
        undo_rotation_action.triggered.connect(self._undo_rotation_all)
        image_adjustments_menu.addAction(undo_rotation_action)

        undo_crop_action = QAction("Undo Crop All", self)
        undo_crop_action.triggered.connect(self._undo_crop_all)
        image_adjustments_menu.addAction(undo_crop_action)

        tools_menu.addSeparator()
        export_images_action = QAction("Export Modified Images", self)
        export_images_action.triggered.connect(self._export_modified_images)
        tools_menu.addAction(export_images_action)

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

        isodata_action = QAction("ISODATA", self)
        isodata_action.triggered.connect(self._run_isodata)
        cluster_menu.addAction(isodata_action)

        analyze_menu = menu_bar.addMenu("Analyze")
        create_hist_action = QAction("Create Histograms", self)
        create_hist_action.triggered.connect(self._create_histograms_from_selection)
        analyze_menu.addAction(create_hist_action)

        create_jointplot_action = QAction("Create Jointplot", self)
        create_jointplot_action.triggered.connect(self._create_jointplot)
        analyze_menu.addAction(create_jointplot_action)

        phenotype_masks_action = QAction("Phenotype Masks", self)
        phenotype_masks_action.triggered.connect(self._phenotype_masks)
        analyze_menu.addAction(phenotype_masks_action)

        mask_menu = menu_bar.addMenu("Mask")
        refine_mask_action = QAction("Refine Mask", self)
        refine_mask_action.triggered.connect(self._refine_mask)
        mask_menu.addAction(refine_mask_action)

        properties_table_action = QAction("Properties Table", self)
        properties_table_action.triggered.connect(self._show_mask_properties)
        mask_menu.addAction(properties_table_action)

        # Create stop button for clustering
        self.stop_btn = QPushButton("Stop Clustering", self)
        self.stop_btn.setStyleSheet("background-color: #ff4d4d; color: white; font-weight: bold; border: none; padding: 5px 10px;")
        self.stop_btn.clicked.connect(self._stop_clustering)
        self.stop_btn.hide()
        
        # Position it in the upper right
        menu_bar.setCornerWidget(self.stop_btn, Qt.TopRightCorner)

    def _stop_clustering(self):
        if hasattr(self, 'clustering_thread') and self.clustering_thread.isRunning():
            reply = QMessageBox.question(self, "Stop Clustering", "Are you sure you want to stop the clustering process?", 
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.Yes:
                self.clustering_thread.terminate()
                self.clustering_thread.wait()
                QApplication.restoreOverrideCursor()
                self.statusBar().showMessage("Clustering stopped by user.")
                if self.stop_btn:
                    self.stop_btn.hide()
        
    def _home_triggered(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Working Directory")
        if directory:
            self._load_working_directory(directory)

    def _load_working_directory(self, directory):
        self.working_dir = directory
        self.asset_manager.set_working_dir(directory)
        
        # Save to settings
        settings = QSettings("CMP", "Viewer")
        settings.setValue("last_working_dir", directory)
        
        # Reconcile project JSON with items on disk
        reconcile_project_with_disk(self.asset_manager, self)
        
        # Check for naming convention violations
        invalid_files = self.asset_manager.validate_filenames()
        if invalid_files:
            msg = "The following files do not match the naming convention:\n\n"
            msg += "\n".join(invalid_files[:20])
            if len(invalid_files) > 20:
                msg += f"\n... and {len(invalid_files) - 20} more."
            msg += "\n\nRequired convention:\n<Sample>_<Slide ##>_<Owner Initials>_<ObjectiveMag>_<Well Position>_<Probe>\n"
            msg += "Example: 123_01_JS_20x_5_DAPI.tif"
            QMessageBox.warning(self, "Naming Convention Warning", msg)

        self.image_handler.clear()
        self.visible_masks.clear()
        self._update_asset_list()
        self._update_mask_list()
        self._update_graph_list()
        self._refresh_viewer()
        self.bg_label.lower()

    def _import_image(self):
        if not self.working_dir:
            QMessageBox.warning(self, "No Working Directory", "Please select a working directory (Home) first.")
            return

        file_path, _ = QFileDialog.getOpenFileName(
            self, "Import Image", "", "Images (*.tif *.tiff *.png *.bmp *.jpg *.jpeg)"
        )

        if file_path:
            import shutil
            dest_path = os.path.join(self.working_dir, os.path.basename(file_path))
            
            if os.path.abspath(file_path) == os.path.abspath(dest_path):
                QMessageBox.information(self, "Import Image", "Image is already in the working directory.")
            else:
                try:
                    shutil.copy(file_path, dest_path)
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"Failed to copy image: {str(e)}")
                    return

            self.asset_manager.scan_assets()
            self._update_asset_list()

    def _import_mask(self):
        if not self.working_dir:
            QMessageBox.warning(self, "No Working Directory", "Please select a working directory (Home) first.")
            return

        file_path, _ = QFileDialog.getOpenFileName(
            self, "Import Mask", "", "Mask Files (*.npy *.png *.jpg *.jpeg *.tif *.tiff *.bmp)"
        )

        if file_path:
            import shutil
            mask_dir = os.path.join(self.working_dir, "Cluster Masks")
            os.makedirs(mask_dir, exist_ok=True)
            dest_path = os.path.join(mask_dir, os.path.basename(file_path))

            if os.path.abspath(file_path) == os.path.abspath(dest_path):
                QMessageBox.information(self, "Import Mask", "Mask is already in the cluster masks directory.")
            else:
                try:
                    shutil.copy(file_path, dest_path)
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"Failed to copy mask: {str(e)}")
                    return

            self.asset_manager.scan_assets()
            self._update_mask_list()

    def _update_asset_list(self):
        # Block signals to prevent _asset_clicked from being triggered during refresh
        self.image_list.blockSignals(True)
        self.image_list.clear()
        for name in self.asset_manager.get_image_list():
            asset = self.asset_manager.get_image_by_name(name)
            item = QListWidgetItem(name)
            qimg = asset.to_qimage()
            if qimg.isNull():
                qimg = QImage(100, 100, QImage.Format_ARGB32)
                qimg.fill(Qt.black)
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
        
        for name in self.asset_manager.get_mask_list():
            asset = self.asset_manager.get_mask_by_name(name)
            item = QListWidgetItem(name)
            qimg = asset.to_qimage()
            if qimg.isNull():
                qimg = QImage(100, 100, QImage.Format_ARGB32)
                qimg.fill(Qt.black)
            
            # Create a composite icon with a color square next to the thumbnail
            thumbnail_size = 50
            thumbnail = qimg.scaled(thumbnail_size, thumbnail_size, Qt.KeepAspectRatio)
            
            # Icon size is 100x50 (50 for thumb, 50 for color square)
            icon_width = 100
            icon_height = 50
            composite = QImage(icon_width, icon_height, QImage.Format_ARGB32)
            composite.fill(Qt.transparent)
            
            painter = QPainter(composite)
            # Draw thumbnail centered in the left half
            x = (thumbnail_size - thumbnail.width()) // 2
            y = (thumbnail_size - thumbnail.height()) // 2
            painter.drawImage(x, y, thumbnail)
            
            # Draw color square in the right half
            color_name = asset.pipeline.config.get("color", "grayscale")
            if color_name == "grayscale":
                qcolor = self.get_mask_color(name)
            else:
                color_rgb = ImageDisplayHandler.COLORS.get(color_name, (1, 1, 1))
                qcolor = QColor(int(color_rgb[0]*255), int(color_rgb[1]*255), int(color_rgb[2]*255))
            
            # Make the square the same size as the thumbnail's bounding box (50x50)
            square_size = 46 # Slightly smaller to have some padding/border visibility
            painter.setBrush(QBrush(qcolor))
            painter.setPen(QPen(Qt.black, 1))
            painter.drawRect(50 + (50 - square_size) // 2, (50 - square_size) // 2, square_size, square_size)
            painter.end()
            
            item.setIcon(QPixmap.fromImage(composite))
            self.mask_list.addItem(item)
            if name in self.visible_masks:
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
            graphs = [f for f in os.listdir(graph_dir) if (f.endswith(".png") and not f.startswith("JointPlot_")) or f.startswith("JointPlot_")]
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
        if not selected_masks:
            QMessageBox.warning(self, "Graphing", "Please select at least one cluster mask.")
            return
        
        mask_names = [item.text() for item in selected_masks]

        selected_images = self.image_list.selectedItems()
        if not selected_images:
            image_names = list(self.image_handler.visible_assets)
        else:
            image_names = [item.text() for item in selected_images]

        if not image_names:
            QMessageBox.warning(self, "Graphing", "No images selected or visible.")
            return

        # Ask for normalization
        options = ["None", "Local (per image)", "Global (entire stack)"]
        item, ok = QInputDialog.getItem(self, "Normalization", 
                                        "Select normalization type (ISODATA style):", 
                                        options, 0, False)
        if not ok:
            return

        normalize = (item == "Local (per image)")
        normalize_stack = (item == "Global (entire stack)")

        try:
            QApplication.setOverrideCursor(Qt.WaitCursor)
            
            # Setup project JSON info
            image_list = self.asset_manager.get_image_list()
            project_name = "project_"
            if image_list:
                parts = image_list[0].split('_')
                if len(parts) >= 2:
                    project_name = f"{parts[0]}_{parts[1]}_"
            
            json_dir = os.path.join(self.working_dir, "JSON")
            project_json_path = os.path.join(json_dir, f"{project_name}.json")
            graph_dir = os.path.join(self.working_dir, "Graphs")

            for i, mask_name in enumerate(mask_names):
                self.statusBar().showMessage(f"Processing mask {i+1}/{len(mask_names)}: {mask_name}...")
                
                measurements = measure_utilities.calculate_mask_measurements(
                    self.asset_manager, image_names, mask_name, 
                    normalize=normalize, normalize_stack=normalize_stack
                )
                if not measurements:
                    continue

                project_data = {}
                if os.path.exists(project_json_path):
                    try:
                        with open(project_json_path, 'r') as f:
                            project_data = json.load(f)
                    except:
                        pass

                # Find mask metadata if available
                mask_metadata = project_data.get("Masks", {}).get(mask_name, {})
                source_masks = mask_metadata.get("source_masks")

                # Recommended behavior: Create PNG, JSON (raw values), and CSV (counts)
                hist_files = histogram_plots.create_histograms(
                    measurements, mask_name, graph_dir, 
                    source_masks=source_masks, show_kde=True,
                    normalization=item
                )
                json_measurements_path = export_plot_utils.save_measurements_json(measurements, mask_name, graph_dir)
                csv_measurements_path = export_plot_utils.save_group_csv(measurements, mask_name, graph_dir)

                if "Histograms" not in project_data:
                    project_data["Histograms"] = {}

                cluster_method = mask_metadata.get("cluster_method", "Unknown")

                for hist_file in hist_files:
                    hist_path = os.path.join(graph_dir, hist_file)
                    
                    stats = {
                        "mean": 0.0, "median": 0.0, "std": 0.0, "skewness": 0.0, "kurtosis": 0.0,
                        "q05": 0.0, "q25": 0.0, "q75": 0.0, "q95": 0.0, "entropy": 0.0
                    }
                    
                    # Match hist_file back to image_name to get values
                    sample_name = "Unknown"
                    slide_number = "Unknown"
                    well_position = "Unknown"
                    probe_name = "Unknown"
                    
                    for img_name, values in measurements.items():
                        from export_plot_utils import get_safe_histogram_name
                        expected_name = f"{get_safe_histogram_name(img_name, mask_name)}.png"
                        
                        if hist_file == expected_name:
                            parts = img_name.split('_')
                            if len(parts) >= 6:
                                sample_name = parts[0]
                                slide_number = parts[1]
                                well_position = parts[4]
                                probe_name = os.path.splitext(parts[5])[0]
                                
                            v = np.array(values)
                            if len(v) > 0:
                                stats["mean"] = float(np.mean(v))
                                stats["median"] = float(np.median(v))
                                stats["std"] = float(np.std(v))
                                stats["skewness"] = float(scipy.stats.skew(v))
                                stats["kurtosis"] = float(scipy.stats.kurtosis(v))
                                stats["q05"] = float(np.percentile(v, 5))
                                stats["q25"] = float(np.percentile(v, 25))
                                stats["q75"] = float(np.percentile(v, 75))
                                stats["q95"] = float(np.percentile(v, 95))
                                
                                try:
                                    counts, _ = np.histogram(v, bins='auto', density=True)
                                    stats["entropy"] = float(scipy.stats.entropy(counts)) if len(counts) > 0 else 0.0
                                except:
                                    stats["entropy"] = 0.0
                            break

                    project_data["Histograms"][hist_file] = {
                        "path": os.path.abspath(hist_path),
                        "sample": sample_name,
                        "slide": slide_number,
                        "well": well_position,
                        "probe": probe_name,
                        "linked_mask": mask_name,
                        "cluster_method": cluster_method,
                        "normalization": item, # Store the normalization type
                        "histograms_json": os.path.abspath(json_measurements_path),
                        "histograms_csv": os.path.abspath(csv_measurements_path),
                        **stats
                    }

                with open(project_json_path, 'w') as f:
                    json.dump(project_data, f, indent=4)

            self.statusBar().showMessage("Graphing completed.", 3000)
            self._update_graph_list()
            self._show_graphs_window()
        except Exception as e:
            QMessageBox.critical(self, "Graphing Error", f"An error occurred: {str(e)}")
        finally:
            QApplication.restoreOverrideCursor()

    def _create_jointplot(self):
        if not self.working_dir:
            QMessageBox.warning(self, "Joint KDE Plot", "Please select a working directory first.")
            return

        # Get list of masks
        mask_dir = os.path.join(self.working_dir, "Cluster Masks")
        if not os.path.exists(mask_dir):
            QMessageBox.warning(self, "Joint KDE Plot", "No masks found in working directory.")
            return
        
        masks = [f for f in os.listdir(mask_dir) if f.endswith(".npy")]
        if not masks:
            QMessageBox.warning(self, "Joint KDE Plot", "No .npy masks found.")
            return

        # Get list of images
        images = self.asset_manager.get_image_list()
        if len(images) < 2:
            QMessageBox.warning(self, "Joint KDE Plot", "Please import at least two images.")
            return

        dialog = JointPlotDialog(masks, images, self)
        if dialog.exec() == QDialog.Accepted:
            selections, user_filename = dialog.get_selections()
            
            # Ask for normalization
            options = ["None", "Local (per image)", "Global (entire stack)"]
            item, ok = QInputDialog.getItem(self, "Normalization", 
                                            "Select normalization type (ISODATA style):", 
                                            options, 0, False)
            if not ok:
                return

            normalize = (item == "Local (per image)")
            normalize_stack = (item == "Global (entire stack)")

            try:
                self.statusBar().showMessage("Generating Joint KDE Plot...")
                QApplication.setOverrideCursor(Qt.WaitCursor)

                all_measurements = []
                for sel in selections:
                    mask_name = sel["mask"]
                    image1_name = sel["image1"]
                    image2_name = sel["image2"]
                    mask_path = os.path.join(mask_dir, mask_name)

                    # Use the same function as histograms to get the data
                    measurements = measure_utilities.calculate_mask_measurements(
                        self.asset_manager, [image1_name, image2_name], mask_path,
                        normalize=normalize, normalize_stack=normalize_stack
                    )
                    if not measurements or len(measurements) < 2:
                        continue
                    
                    x_values = measurements[image1_name]
                    y_values = measurements[image2_name]
                    
                    # Ensure same length
                    min_len = min(len(x_values), len(y_values))
                    x_values = x_values[:min_len]
                    y_values = y_values[:min_len]

                    all_measurements.append({
                        "image1_name": image1_name,
                        "image2_name": image2_name,
                        "mask_name": mask_name,
                        "x_values": x_values,
                        "y_values": y_values
                    })

                if not all_measurements:
                    QMessageBox.critical(self, "Joint KDE Plot", "Failed to calculate histogram data for any of the selected sets.")
                    return

                graph_dir = os.path.join(self.working_dir, "Graphs")
                filename = kde_plots.create_joint_kde_plot(all_measurements, graph_dir, user_filename=user_filename, normalization=item)

                if filename:
                    self.statusBar().showMessage("Joint KDE Plot completed.", 3000)
                    self._update_graph_list()
                    # Select the newly created graph
                    items = self.graph_list.findItems(filename, Qt.MatchExactly)
                    if items:
                        self.graph_list.clearSelection()
                        items[0].setSelected(True)
                    self._show_graphs_window()
                else:
                    QMessageBox.warning(self, "Joint KDE Plot", "Could not create plot.")

            except Exception as e:
                QMessageBox.critical(self, "Joint KDE Plot Error", f"An error occurred: {str(e)}")
            finally:
                QApplication.restoreOverrideCursor()

    def _phenotype_masks(self):
        if not hasattr(self, 'asset_manager') or not self.asset_manager.working_dir:
            QMessageBox.warning(self, "Phenotype Masks", "Please select a working directory first.")
            return
            
        selected_items = self.mask_list.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "No Selection", "Please select one or more masks to phenotype.")
            return
            
        selected_masks = [item.text() for item in selected_items]
        
        # Check if images are also selected
        selected_images = self.image_list.selectedItems()
        image_names = [item.text() for item in selected_images]
        
        normalization_type = "None"
        if image_names:
            options = ["None", "Local (per image)", "Global (entire stack)"]
            item, ok = QInputDialog.getItem(self, "Normalization", 
                                            "Select normalization type for histograms:", 
                                            options, 0, False)
            if not ok:
                return
            normalization_type = item

        # Prompt user to select tissue area mask
        all_masks = self.asset_manager.get_mask_list()
        if not all_masks:
            QMessageBox.warning(self, "Phenotype Masks", "No masks found in the project to use as tissue area.")
            return

        tissue_mask, ok = QInputDialog.getItem(self, "Select Tissue Area Mask", 
                                              "Select the mask that represents the tissue area:", 
                                              all_masks, 0, False)
        if not ok:
            return

        tissue_type, ok = QInputDialog.getItem(self, "Select Tissue Type", 
                                              "Select the tissue type:", 
                                              ["Retina", "Organoid"], 0, False)
        if not ok:
            return

        output_dir = os.path.join(self.asset_manager.working_dir, "Phenotypes")
        
        try:
            self.statusBar().showMessage("Phenotyping masks...")
            QApplication.setOverrideCursor(Qt.WaitCursor)
            
            import mask_phenotyping
            
            # Determine project name for filename
            image_list = self.asset_manager.get_image_list()
            project_name = None
            if image_list:
                parts = image_list[0].split('_')
                if len(parts) >= 2:
                    project_name = f"{parts[0]}_{parts[1]}"
            
            csv_paths = mask_phenotyping.phenotype_masks(
                self.asset_manager, selected_masks, output_dir, tissue_mask, tissue_type,
                image_names=image_names, normalization_type=normalization_type,
                project_name=project_name
            )
            
            if csv_paths:
                # Update project JSON
                for path in csv_paths:
                    self.asset_manager.add_phenotype_reference(path)
                    
                self.statusBar().showMessage(f"Phenotyping complete. Data saved to {output_dir}", 5000)
                paths_str = "\n".join(csv_paths)
                QMessageBox.information(self, "Success", f"Phenotyping complete.\n\nResults saved to:\n{paths_str}")
            else:
                QMessageBox.warning(self, "Error", "Phenotyping failed or no objects found in the selected masks.")
                
        except Exception as e:
            QMessageBox.critical(self, "Phenotyping Error", f"An error occurred during phenotyping:\n{str(e)}")
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
            widget = self.graphs_layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)

        selected_graphs = self.graph_list.selectedItems()
        if not selected_graphs:
            return
        
        graph_dir = os.path.join(self.working_dir, "Graphs")
        show_kde = self.kde_checkbox.isChecked()
        
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
            # If high quality is requested, we use raw values from JSON
            # Otherwise we use counts from CSV
            
            # Find project JSON path
            image_list = self.asset_manager.get_image_list()
            project_name = "project_"
            if image_list:
                parts = image_list[0].split('_')
                if len(parts) >= 2:
                    project_name = f"{parts[0]}_{parts[1]}_"
            
            json_dir = os.path.join(self.working_dir, "JSON")
            project_json_path = os.path.join(json_dir, f"{project_name}.json")

            project_data = {}
            if os.path.exists(project_json_path):
                try:
                    with open(project_json_path, 'r') as f:
                        project_data = json.load(f)
                except:
                    pass

            source_masks = None
            items_to_render = [] # Either (label, values) or (label, counts)

            if show_kde:
                # High quality - Load JSON raw values
                mask_to_measurements = {}
                if os.path.exists(graph_dir):
                    for f in os.listdir(graph_dir):
                        if f.startswith("Histograms_") and f.endswith(".json"):
                            mask_name_from_json = f[len("Histograms_"):-len(".json")]
                            try:
                                with open(os.path.join(graph_dir, f), 'r') as jf:
                                    mask_to_measurements[mask_name_from_json] = json.load(jf)
                            except:
                                pass

                for item in selected_graphs:
                    name = item.text()
                    if name.startswith("JointPlot_"): continue
                    
                    found = False
                    for mask_name, measurements in mask_to_measurements.items():
                        if mask_name in name:
                            source_masks = project_data.get("Masks", {}).get(mask_name, {}).get("source_masks")
                            for img_name, values in measurements.items():
                                from export_plot_utils import get_safe_histogram_name
                                column_header = get_safe_histogram_name(img_name, mask_name)
                                if column_header in name:
                                    items_to_render.append((f"{img_name} ({mask_name})", values))
                                    found = True
                                    break
                            if found: break
                
                if items_to_render:
                    combined_rgb = histogram_plots.create_dynamic_overlaid_histogram(items_to_render, source_masks=source_masks, show_kde=True)
            else:
                # Fast rendering - Load CSV counts
                mask_to_counts = {}
                if os.path.exists(graph_dir):
                    for f in os.listdir(graph_dir):
                        if f.startswith("Histograms_") and f.endswith(".csv"):
                            mask_name_from_csv = f[len("Histograms_"):-len(".csv")]
                            try:
                                df = pd.read_csv(os.path.join(graph_dir, f))
                                mask_to_counts[mask_name_from_csv] = df
                            except:
                                pass
                
                for item in selected_graphs:
                    name = item.text()
                    if name.startswith("JointPlot_"): continue
                    
                    found = False
                    # Remove .png extension for matching with CSV headers
                    name_no_ext = os.path.splitext(name)[0]
                    
                    for mask_name, df in mask_to_counts.items():
                        if mask_name in name:
                            source_masks = project_data.get("Masks", {}).get(mask_name, {}).get("source_masks")
                            if name_no_ext in df.columns:
                                counts = df[name_no_ext].values
                                items_to_render.append((name_no_ext, counts))
                                found = True
                        if found: break
                
                if items_to_render:
                    combined_rgb = histogram_plots.render_fast_overlay(items_to_render, source_masks=source_masks)

            if items_to_render and combined_rgb is not None:
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
            QMessageBox.warning(self, "Save PNG", "No graphs selected.")
            return

        path, _ = QFileDialog.getSaveFileName(self, "Save Combined Histogram", "", "PNG (*.png)")
        if not path: return

        graph_dir = os.path.join(self.working_dir, "Graphs")
        # Get project JSON
        image_list = self.asset_manager.get_image_list()
        project_name = "project_"
        if image_list:
            parts = image_list[0].split('_')
            if len(parts) >= 2:
                project_name = f"{parts[0]}_{parts[1]}_"
        
        json_dir = os.path.join(self.working_dir, "JSON")
        project_json_path = os.path.join(json_dir, f"{project_name}.json")

        project_data = {}
        if os.path.exists(project_json_path):
            try:
                with open(project_json_path, 'r') as f:
                    project_data = json.load(f)
            except:
                pass

        if len(selected) == 1:
            # Just copy the file
            import shutil
            shutil.copy(os.path.join(graph_dir, selected[0].text()), path)
            QMessageBox.information(self, "Save PNG", f"Graph saved to {path}")
        else:
            # Create and save combined - ALWAYS HIGH QUALITY
            items_measurements = []
            
            mask_to_measurements = {}
            if os.path.exists(graph_dir):
                for f in os.listdir(graph_dir):
                    if f.startswith("Histograms_") and f.endswith(".json"):
                        mask_name_from_json = f[len("Histograms_"):-len(".json")]
                        try:
                            with open(os.path.join(graph_dir, f), 'r') as jf:
                                mask_to_measurements[mask_name_from_json] = json.load(jf)
                        except:
                            pass

            source_masks_list = []
            for item in selected:
                name = item.text()
                if name.startswith("JointPlot_"):
                    continue

                found = False
                for mask_name, measurements in mask_to_measurements.items():
                    if mask_name in name:
                        # Track source masks if any
                        s_masks = project_data.get("Masks", {}).get(mask_name, {}).get("source_masks")
                        if s_masks:
                            source_masks_list.extend(s_masks)

                        for img_name, values in measurements.items():
                            # Use the new naming convention to match the graph name
                            from export_plot_utils import get_safe_histogram_name
                            column_header = get_safe_histogram_name(img_name, mask_name)
                        
                            if column_header in name:
                                items_measurements.append((column_header, values))
                                found = True
                                break
                        if found: break
                
            if items_measurements:
                # Remove duplicates from source_masks_list
                unique_sources = sorted(list(set(source_masks_list)))
                histogram_plots.create_dynamic_overlaid_histogram(items_measurements, output_path=path, source_masks=unique_sources if unique_sources else None, show_kde=True)
                QMessageBox.information(self, "Save PNG", f"Combined high-quality histogram saved to {path}")

    def _export_selected_graphs(self):
        if not self.working_dir: return
        selected = self.graph_list.selectedItems()
        if not selected:
            QMessageBox.warning(self, "Export CSV", "No graphs selected.")
            return

        path, _ = QFileDialog.getSaveFileName(self, "Export Raw Values", "", "CSV (*.csv)")
        if not path: return

        graph_dir = os.path.join(self.working_dir, "Graphs")
        items_measurements = []
        
        mask_to_measurements = {}
        if os.path.exists(graph_dir):
            for f in os.listdir(graph_dir):
                if f.startswith("Histograms_") and f.endswith(".json"):
                    mask_name_from_json = f[len("Histograms_"):-len(".json")]
                    try:
                        with open(os.path.join(graph_dir, f), 'r') as jf:
                            mask_to_measurements[mask_name_from_json] = json.load(jf)
                    except:
                        pass

        for item in selected:
            name = item.text()
            if name.startswith("JointPlot_"):
                continue

            found = False
            for mask_name, measurements in mask_to_measurements.items():
                if mask_name in name:
                    for img_name, values in measurements.items():
                        # Use the new naming convention to match the graph name
                        from export_plot_utils import get_safe_histogram_name
                        column_header = get_safe_histogram_name(img_name, mask_name)
                        
                        if column_header in name:
                            items_measurements.append((column_header, values))
                            found = True
                            break
                    if found: break

        if not items_measurements:
            QMessageBox.warning(self, "Export CSV", "Could not find raw data for selected graphs.")
            return

        try:
            import csv
            # We want to export histogram frequencies (0-255 intensity).
            
            with open(path, 'w', newline='') as f:
                writer = csv.writer(f)
                # Header: Intensity, then all selected labels
                labels = [label for label, _ in items_measurements]
                writer.writerow(["Intensity"] + labels)
                
                # Calculate histograms for each
                histograms = []
                for _, values in items_measurements:
                    if len(values) > 0:
                        v = np.array(values)
                        if v.max() <= 1.0001 and v.min() >= -0.0001:
                            # If values are normalized (0-1), scale to 0-255 for the fixed-bin CSV export
                            counts, _ = np.histogram(v * 255.0, bins=range(257))
                        else:
                            counts, _ = np.histogram(v, bins=range(257))
                        histograms.append(counts)
                    else:
                        histograms.append([0] * 256)
                
                # Data rows (0-255)
                for i in range(256):
                    row = [i]
                    for h in histograms:
                        row.append(h[i])
                    writer.writerow(row)
            
            QMessageBox.information(self, "Export CSV", f"Histogram data exported to {path}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"An error occurred: {str(e)}")

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
                
                # Move to deleted assets in JSON
                self.asset_manager.move_to_deleted_assets(name, "Histogram")

                path = os.path.join(graph_dir, name)
                if os.path.exists(path):
                    os.remove(path)
            self._update_graph_list()
            self._show_graphs_window()

    def _on_clustering_finished(self, cluster_mask, mask_root_name, n_clusters, stats_csv_path, algorithm=None, params=None, image_names=None):
        try:
            mask_dir = os.path.join(self.working_dir, "Cluster Masks")
            os.makedirs(mask_dir, exist_ok=True)

            individual_masks = clustering.get_individual_masks(cluster_mask, n_clusters)

            # Prepare metadata for linking in project JSON
            # Image info: <Well Position>_<Probe>
            source_images_info = []
            if image_names:
                for img_name in image_names:
                    parts = img_name.split('_')
                    if len(parts) >= 6:
                        # <Sample>_<Slide ##>_<Owner Initials>_<ObjectiveMag>_<Well Position>_<Probe>
                        # Indices: 0, 1, 2, 3, 4, 5
                        # Strip extension from the last part
                        probe = os.path.splitext(parts[5])[0]
                        source_images_info.append(f"{parts[4]}_{probe}")
                    else:
                        source_images_info.append(img_name)

            # Get project JSON path
            image_list = self.asset_manager.get_image_list()
            project_name = "project_"
            if image_list:
                parts = image_list[0].split('_')
                if len(parts) >= 2:
                    project_name = f"{parts[0]}_{parts[1]}_"
            
            json_dir = os.path.join(self.working_dir, "JSON")
            project_json_path = os.path.join(json_dir, f"{project_name}.json")

            project_data = {}
            if os.path.exists(project_json_path):
                try:
                    with open(project_json_path, 'r') as f:
                        project_data = json.load(f)
                except:
                    pass
            
            if "Masks" not in project_data:
                project_data["Masks"] = {}

            for i, mask in enumerate(individual_masks):
                new_mask_name = f"{mask_root_name}_{i+1:02d}.png"
                # Save mask as 8-bit PNG
                mask_to_save = mask.astype(np.uint8)
                cv2.imwrite(os.path.join(mask_dir, new_mask_name), mask_to_save)

                # Link in project JSON
                mask_path = os.path.join(mask_dir, new_mask_name)
                project_data["Masks"][new_mask_name] = {
                    "path": os.path.abspath(mask_path),
                    "source_images": source_images_info,
                    "cluster_method": algorithm,
                    "cluster_parameters": params,
                    "statistics_csv": os.path.abspath(stats_csv_path) if stats_csv_path else None
                }

            with open(project_json_path, 'w') as f:
                json.dump(project_data, f, indent=4)

            msg = "Clustering completed."
            if stats_csv_path:
                msg += f" Statistics saved to {os.path.basename(stats_csv_path)}."
            self.statusBar().showMessage(msg, 5000)
            self.asset_manager.scan_assets()
            self._update_mask_list()
            self._update_graph_list()
        except Exception as e:
            QMessageBox.critical(self, "Clustering Error", f"An error occurred while saving results: {str(e)}")
        finally:
            if self.stop_btn:
                self.stop_btn.hide()
            QApplication.restoreOverrideCursor()
            if hasattr(self, 'clustering_thread'):
                self.clustering_thread.quit()
                self.clustering_thread.wait()

    def _on_clustering_error(self, error_msg):
        if self.stop_btn:
            self.stop_btn.hide()
        QApplication.restoreOverrideCursor()
        QMessageBox.critical(self, "Clustering Error", f"An error occurred during clustering: {error_msg}")
        if hasattr(self, 'clustering_thread'):
            self.clustering_thread.quit()
            self.clustering_thread.wait()

    def _run_kmeans(self):
        if hasattr(self, 'clustering_thread') and self.clustering_thread.isRunning():
            QMessageBox.warning(self, "K-Means", "A clustering process is already running. Please wait for it to finish.")
            return

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

        # Get available masks
        mask_names = []
        mask_dir = os.path.join(self.working_dir, "Cluster Masks")
        if os.path.exists(mask_dir):
            mask_names = [f for f in os.listdir(mask_dir) if f.lower().endswith((".png", ".npy"))]
            mask_names.sort()

        dialog = ClusterParameterDialog(mask_names, self)
        if dialog.exec() == QDialog.Accepted:
            params = dialog.get_params()
            mask_to_use = None
            mask_name = params.pop("mask_name", "None")
            if mask_name != "None":
                mask_root_name = os.path.splitext(mask_name)[0]
                mask_asset = self.asset_manager.get_mask_by_name(mask_name)
                if mask_asset:
                    mask_to_use = mask_asset.get_rendered_data(data_only=True)
                else:
                    QMessageBox.warning(self, "K-Means", f"Failed to load mask: {mask_name}")
            else:
                mask_root_name = "KM"
            
            try:
                if self.stop_btn:
                    self.stop_btn.show()
                self.statusBar().showMessage("Running K-Means...")
                QApplication.setOverrideCursor(Qt.WaitCursor)
                
                graph_dir = os.path.join(self.working_dir, "Graphs")
                os.makedirs(graph_dir, exist_ok=True)

                self.clustering_thread = QThread()
                self.clustering_worker = ClusteringWorker(
                    "kmeans", stack, mask_to_use, params, mask_root_name, 
                    image_names=image_names, output_dir=graph_dir
                )
                self.clustering_worker.moveToThread(self.clustering_thread)
                
                self.clustering_thread.started.connect(self.clustering_worker.run)
                self.clustering_worker.finished.connect(
                    lambda mask, root, n, stats: self._on_clustering_finished(
                        mask, root, n, stats, algorithm="kmeans", params=params, image_names=image_names
                    )
                )
                self.clustering_worker.error.connect(self._on_clustering_error)
                
                self.clustering_thread.start()
            except Exception as e:
                QMessageBox.critical(self, "K-Means Error", f"An error occurred while starting clustering: {str(e)}")
                QApplication.restoreOverrideCursor()

    def _run_gmm(self):
        if hasattr(self, 'clustering_thread') and self.clustering_thread.isRunning():
            QMessageBox.warning(self, "Gaussian Mixture", "A clustering process is already running. Please wait for it to finish.")
            return

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

        # Get available masks
        mask_names = []
        mask_dir = os.path.join(self.working_dir, "Cluster Masks")
        if os.path.exists(mask_dir):
            mask_names = [f for f in os.listdir(mask_dir) if f.lower().endswith((".png", ".npy"))]
            mask_names.sort()

        dialog = GMMParameterDialog(mask_names, self)
        if dialog.exec() == QDialog.Accepted:
            params = dialog.get_params()
            mask_to_use = None
            mask_name = params.pop("mask_name", "None")
            if mask_name != "None":
                mask_root_name = os.path.splitext(mask_name)[0]
                mask_asset = self.asset_manager.get_mask_by_name(mask_name)
                if mask_asset:
                    mask_to_use = mask_asset.get_rendered_data(data_only=True)
                else:
                    QMessageBox.warning(self, "Gaussian Mixture", f"Failed to load mask: {mask_name}")
            else:
                mask_root_name = "GM"

            try:
                if self.stop_btn:
                    self.stop_btn.show()
                self.statusBar().showMessage("Running Gaussian Mixture...")
                QApplication.setOverrideCursor(Qt.WaitCursor)
                
                graph_dir = os.path.join(self.working_dir, "Graphs")
                os.makedirs(graph_dir, exist_ok=True)

                self.clustering_thread = QThread()
                self.clustering_worker = ClusteringWorker(
                    "gmm", stack, mask_to_use, params, mask_root_name, 
                    image_names=image_names, output_dir=graph_dir
                )
                self.clustering_worker.moveToThread(self.clustering_thread)
                
                self.clustering_thread.started.connect(self.clustering_worker.run)
                self.clustering_worker.finished.connect(
                    lambda mask, root, n, stats: self._on_clustering_finished(
                        mask, root, n, stats, algorithm="gmm", params=params, image_names=image_names
                    )
                )
                self.clustering_worker.error.connect(self._on_clustering_error)
                
                self.clustering_thread.start()
            except Exception as e:
                QMessageBox.critical(self, "GMM Error", f"An error occurred while starting clustering: {str(e)}")
                QApplication.restoreOverrideCursor()

    def _run_isodata(self):
        if hasattr(self, 'clustering_thread') and self.clustering_thread.isRunning():
            QMessageBox.warning(self, "ISODATA", "A clustering process is already running. Please wait for it to finish.")
            return

        if not self.working_dir:
            QMessageBox.warning(self, "ISODATA", "Please select a working directory first.")
            return

        selected_items = self.image_list.selectedItems()
        if not selected_items:
            image_names = list(self.image_handler.visible_assets)
        else:
            image_names = [item.text() for item in selected_items]

        if not image_names:
            QMessageBox.warning(self, "ISODATA", "No images selected or visible to cluster.")
            return

        stack = image_stacker.load_and_stack_images(self.asset_manager, image_names)
        if stack is None:
            QMessageBox.critical(self, "ISODATA", "Failed to create image stack.")
            return

        # Ensure stack is float32
        stack = stack.astype(np.float32)

        # Get available masks
        mask_names = []
        mask_dir = os.path.join(self.working_dir, "Cluster Masks")
        if os.path.exists(mask_dir):
            mask_names = [f for f in os.listdir(mask_dir) if f.lower().endswith((".png", ".npy"))]
            mask_names.sort()

        dialog = ISODATAParameterDialog(mask_names, self)
        if dialog.exec() == QDialog.Accepted:
            params = dialog.get_params()
            mask_to_use = None
            mask_name = params.pop("mask_name", "None")
            if mask_name != "None":
                mask_root_name = os.path.splitext(mask_name)[0]
                mask_asset = self.asset_manager.get_mask_by_name(mask_name)
                if mask_asset:
                    mask_to_use = mask_asset.get_rendered_data(data_only=True)
                else:
                    QMessageBox.warning(self, "ISODATA", f"Failed to load mask: {mask_name}")
            else:
                mask_root_name = "IS"

            try:
                if self.stop_btn:
                    self.stop_btn.show()
                self.statusBar().showMessage("Running ISODATA...")
                QApplication.setOverrideCursor(Qt.WaitCursor)
                
                graph_dir = os.path.join(self.working_dir, "Graphs")
                os.makedirs(graph_dir, exist_ok=True)

                self.clustering_thread = QThread()
                self.clustering_worker = ClusteringWorker(
                    "isodata", stack, mask_to_use, params, mask_root_name,
                    image_names=image_names, output_dir=graph_dir
                )
                self.clustering_worker.moveToThread(self.clustering_thread)

                self.clustering_thread.started.connect(self.clustering_worker.run)
                self.clustering_worker.finished.connect(
                    lambda mask, root, n, stats: self._on_clustering_finished(
                        mask, root, n, stats, algorithm="isodata", params=params, image_names=image_names
                    )
                )
                self.clustering_worker.error.connect(self._on_clustering_error)

                self.clustering_thread.start()
            except Exception as e:
                QMessageBox.critical(self, "ISODATA Error", f"An error occurred while starting clustering: {str(e)}")
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
            self.preview_mask = mask_refinement.create_threshold_mask(stack, params["threshold"], params["normalize"])
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
                
                binary_mask = mask_refinement.create_threshold_mask(stack, params["threshold"], params["normalize"])

                mask_dir = os.path.join(self.working_dir, "Cluster Masks")
                os.makedirs(mask_dir, exist_ok=True)

                # Find a unique name for the threshold mask
                idx = 1
                while os.path.exists(os.path.join(mask_dir, f"ThresholdMask_{idx}.png")):
                    idx += 1
                mask_name = f"ThresholdMask_{idx}.png"
                cv2.imwrite(os.path.join(mask_dir, mask_name), binary_mask.astype(np.uint8))

                self.statusBar().showMessage(f"Threshold mask {mask_name} created.", 3000)
                self.asset_manager.scan_assets()
                self._update_mask_list()
            except Exception as e:
                QMessageBox.critical(self, "Threshold Error", f"An error occurred: {str(e)}")
            finally:
                QApplication.restoreOverrideCursor()
        
        self.preview_mask = None
        self.cached_composite = None
        self._refresh_viewer()

    def _refine_mask(self):
        if not self.working_dir:
            QMessageBox.warning(self, "Refine Mask", "Please select a working directory first.")
            return

        selected_items = self.mask_list.selectedItems()
        if len(selected_items) != 1:
            QMessageBox.warning(self, "Refine Mask", "Please select exactly one mask to refine.")
            return

        mask_name = selected_items[0].text()
        mask_asset = self.asset_manager.get_mask_by_name(mask_name)
        if not mask_asset:
            return

        try:
            mask = mask_asset.get_rendered_data(data_only=True)
            if mask is None:
                raise ValueError("Failed to load mask data.")
        except Exception as e:
            QMessageBox.critical(self, "Refine Mask", f"Failed to load mask: {str(e)}")
            return

        dialog = RefineMaskDialog(self)
        if dialog.exec() == QDialog.Accepted:
            params = dialog.get_params()
            try:
                self.statusBar().showMessage("Refining mask...")
                QApplication.setOverrideCursor(Qt.WaitCursor)
                
                refined_mask = mask_refinement.refine_mask(
                    mask, 
                    min_area=params["min_area"], 
                    min_circularity=params["min_circularity"], 
                    max_eccentricity=params["max_eccentricity"],
                    min_solidity=params["min_solidity"],
                    min_extent=params["min_extent"],
                    euler_number=params["euler_number"]
                )
                
                # Save the new mask
                mask_dir = os.path.join(self.working_dir, "Cluster Masks")
                base_name = os.path.splitext(mask_name)[0]
                refined_mask_name = f"{base_name}_refined.png"
                refined_mask_path = os.path.join(mask_dir, refined_mask_name)
                
                cv2.imwrite(refined_mask_path, refined_mask.astype(np.uint8))
                
                self.asset_manager.scan_assets()
                self._update_mask_list()
                self.statusBar().showMessage(f"Refined mask {refined_mask_name} saved.", 3000)
                QMessageBox.information(self, "Refine Mask", f"Refined mask saved as {refined_mask_name}")
            except Exception as e:
                QMessageBox.critical(self, "Refine Mask", f"Error during refinement: {str(e)}")
            finally:
                QApplication.restoreOverrideCursor()

    def _show_mask_properties(self):
        if not self.working_dir:
            QMessageBox.warning(self, "Properties Table", "Please select a working directory first.")
            return

        selected_items = self.mask_list.selectedItems()
        if len(selected_items) != 1:
            QMessageBox.warning(self, "Properties Table", "Please select exactly one mask to view properties.")
            return

        mask_name = selected_items[0].text()
        mask_asset = self.asset_manager.get_mask_by_name(mask_name)
        if not mask_asset:
            return

        try:
            mask = mask_asset.get_rendered_data(data_only=True)
            if mask is None:
                raise ValueError("Failed to load mask data.")
            properties = mask_refinement.get_mask_properties(mask)
            
            if not properties:
                QMessageBox.information(self, "Properties Table", "No objects found in the selected mask.")
                return

            dialog = MaskPropertiesDialog(properties, mask_name, self)
            dialog.exec()
        except Exception as e:
            QMessageBox.critical(self, "Properties Table", f"Error extracting properties: {str(e)}")

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
        # Invalidate overlay cache when opacity changes
        self.image_handler._cached_mask_overlay = None 
        self.cached_composite = None
        self._refresh_viewer()

    def _show_image_context_menu(self, position):
        item = self.image_list.itemAt(position)
        if not item: return
        self._show_context_menu(self.image_list, position, False)

    def _show_mask_context_menu(self, position):
        item = self.mask_list.itemAt(position)
        if not item: return
        name = item.text()
        
        menu = QMenu()
        
        # Color submenu
        color_menu = menu.addMenu("Color")
        for color_name in ["grayscale", "red", "green", "blue", "cyan", "magenta", "yellow", "white"]:
            action = QAction(color_name.capitalize(), self)
            action.triggered.connect(lambda checked=False, c=color_name: self._change_color(name, c, is_mask=True))
            color_menu.addAction(action)

        mask_asset = self.asset_manager.get_mask_by_name(name)
        if mask_asset:
            invert_action = QAction("Invert", self)
            invert_action.setCheckable(True)
            invert_action.setChecked(mask_asset.pipeline.config.get("invert", False))
            invert_action.triggered.connect(lambda: self._toggle_transform(mask_asset, "invert"))
            menu.addAction(invert_action)

        rename_action = QAction("Rename", self)
        rename_action.triggered.connect(lambda: self._rename_item(item, "Cluster Masks", self.mask_list))
        menu.addAction(rename_action)
        
        menu.exec(self.mask_list.mapToGlobal(position))

    def _show_graph_context_menu(self, position):
        item = self.graph_list.itemAt(position)
        if not item: return
        
        menu = QMenu()
        rename_action = QAction("Rename", self)
        rename_action.triggered.connect(lambda: self._rename_item(item, "Graphs", self.graph_list))
        menu.addAction(rename_action)
        menu.exec(self.graph_list.mapToGlobal(position))

    def _rename_item(self, item, subfolder, list_widget):
        if not self.working_dir: return
        
        old_name = item.text()
        base_name, extension = os.path.splitext(old_name)
        
        new_name, ok = QInputDialog.getText(self, "Rename Item", "Enter new name:", QLineEdit.Normal, base_name)
        
        if ok and new_name and new_name != base_name:
            new_filename = new_name + extension
            old_path = os.path.join(self.working_dir, subfolder, old_name)
            new_path = os.path.join(self.working_dir, subfolder, new_filename)
            
            if os.path.exists(new_path):
                QMessageBox.warning(self, "Rename Error", f"A file with the name '{new_filename}' already exists.")
                return
                
            try:
                # Handle JSON renaming first
                asset = None
                if subfolder == "Cluster Masks":
                    asset = self.asset_manager.get_mask_by_name(old_name)
                elif subfolder == "Images":
                    asset = self.asset_manager.get_image_by_name(old_name)
                
                if asset:
                    old_json = asset.get_json_path()
                    if os.path.exists(old_json):
                        # Temporarily change attributes to get new json path
                        orig_path = asset.path
                        orig_base = asset.base_name
                        orig_name = asset.name
                        
                        asset.path = new_path
                        asset.base_name = new_filename
                        asset.name = new_filename
                        
                        new_json = asset.get_json_path()
                        
                        # Revert for now, will set permanently after successful rename
                        asset.path = orig_path
                        asset.base_name = orig_base
                        asset.name = orig_name
                        
                        if os.path.exists(new_json):
                             os.remove(new_json) # Overwrite if exists
                        os.rename(old_json, new_json)

                os.rename(old_path, new_path)
                
        # Update tracking if necessary (e.g., visible masks)
                if subfolder == "Cluster Masks":
                    if old_name in self.visible_masks:
                        self.visible_masks.remove(old_name)
                        self.visible_masks.add(new_filename)
                    
                    self.image_handler.rename_asset(old_name, new_filename)
                    
                    # Update the asset's internal knowledge of its name/path if it's already loaded
                    if asset:
                        asset.path = new_path
                        asset.base_name = new_filename
                        asset.name = new_filename
                        # Rename in asset_manager maps
                        del self.asset_manager.masks[old_name]
                        self.asset_manager.masks[new_filename] = asset
                    
                    self._update_mask_list()
                    self.cached_composite = None
                    self._refresh_viewer()
                elif subfolder == "Graphs":
                    self._update_graph_list()
                elif subfolder == "Images":
                    self.image_handler.rename_asset(old_name, new_filename)
                    if asset:
                        asset.path = new_path
                        asset.base_name = new_filename
                        asset.name = new_filename
                        del self.asset_manager.images[old_name]
                        self.asset_manager.images[new_filename] = asset
                    self._update_asset_list()
                    self.cached_composite = None
                    self._refresh_viewer()
                    
            except Exception as e:
                QMessageBox.critical(self, "Rename Error", f"Failed to rename file: {str(e)}")

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


    def _undo_invert_all(self):
        if not self.asset_manager.images:
            return
            
        reply = QMessageBox.warning(self, "Undo for All", 
                                  "This will UNDO inversion for ALL loaded images. Do you want to continue?",
                                  QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.No:
            return

        for name, asset in self.asset_manager.images.items():
            asset.pipeline.config["invert"] = False
            asset.save_project()
        
        for name, asset in self.asset_manager.masks.items():
            asset.pipeline.config["invert"] = False
            asset.save_project()
        
        self.cached_composite = None
        self._update_asset_list()
        self._update_mask_list()
        self._refresh_viewer()

    def _undo_rotation_all(self):
        if not self.asset_manager.images:
            return
            
        reply = QMessageBox.warning(self, "Undo for All", 
                                  "This will REMOVE rotation from ALL loaded images. Do you want to continue?",
                                  QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.No:
            return

        for name, asset in self.asset_manager.images.items():
            transforms = asset.pipeline.config.get("transforms", [])
            asset.pipeline.config["transforms"] = [t for t in transforms if t.get("type") != "rotate"]
            asset.save_project()
        
        for name, asset in self.asset_manager.masks.items():
            transforms = asset.pipeline.config.get("transforms", [])
            asset.pipeline.config["transforms"] = [t for t in transforms if t.get("type") != "rotate"]
            asset.save_project()
        
        self.cached_composite = None
        self._update_asset_list()
        self._update_mask_list()
        self._refresh_viewer()

    def _undo_crop_all(self):
        if not self.asset_manager.images:
            return
            
        reply = QMessageBox.warning(self, "Undo for All", 
                                  "This will REMOVE crop from ALL loaded images. Do you want to continue?",
                                  QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.No:
            return

        for name, asset in self.asset_manager.images.items():
            transforms = asset.pipeline.config.get("transforms", [])
            asset.pipeline.config["transforms"] = [t for t in transforms if t.get("type") != "crop"]
            asset.save_project()
        
        for name, asset in self.asset_manager.masks.items():
            transforms = asset.pipeline.config.get("transforms", [])
            asset.pipeline.config["transforms"] = [t for t in transforms if t.get("type") != "crop"]
            asset.save_project()
        
        self.cached_composite = None
        self._update_asset_list()
        self._update_mask_list()
        self._refresh_viewer()


    def _change_color(self, name, color_name, is_mask=False):
        if is_mask:
            self.image_handler.set_asset_color(name, color_name)
            asset = self.asset_manager.get_mask_by_name(name)
            if asset:
                asset.pipeline.config["color"] = color_name
                asset.save_project()
            # Invalidate mask overlay cache
            self.image_handler._cached_mask_overlay = None
            self.cached_composite = None
            self._update_mask_list()
            self._refresh_viewer()
            return

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
        # If it's a mask asset, invalidate mask overlay cache
        from assets import MaskAsset
        if isinstance(asset, MaskAsset):
            self.image_handler._cached_mask_overlay = None
        self.cached_composite = None
        self._update_asset_list()
        self._refresh_viewer()

    def _invert_all_images(self):
        if not self.asset_manager.images:
            return
            
        reply = QMessageBox.warning(self, "Apply to All", 
                                  "This operation will be applied to ALL loaded images. Do you want to continue?",
                                  QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.No:
            return

        for name, asset in self.asset_manager.images.items():
            asset.pipeline.config["invert"] = not asset.pipeline.config.get("invert", False)
            asset.save_project()
        
        for name, asset in self.asset_manager.masks.items():
            asset.pipeline.config["invert"] = not asset.pipeline.config.get("invert", False)
            asset.save_project()
        
        # Invalidate mask overlay cache
        self.image_handler._cached_mask_overlay = None
            
        self.cached_composite = None
        self._update_asset_list()
        self._update_mask_list()
        self._refresh_viewer()

    def _apply_filter_to_all(self, filter_name):
        if not self.asset_manager.images:
            return
            
        reply = QMessageBox.warning(self, "Apply to All", 
                                  f"The {filter_name} filter will be applied to ALL loaded images. Do you want to continue?",
                                  QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.No:
            return
        
        # Use the first image's params as initial if available
        first_asset = list(self.asset_manager.images.values())[0]
        initial_params = first_asset.pipeline.config.get("filter_params", {}).get(filter_name, {})
        
        dialog = FilterParameterDialog(filter_name, initial_params, self)
        if dialog.exec() == QDialog.Accepted:
            params = dialog.get_params()
            for name, asset in self.asset_manager.images.items():
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
        if not self.asset_manager.images:
            return

        rect = self.viewer_view.get_selection_rect()
        if not rect:
            QMessageBox.warning(self, "No Selection", "Please create a selection rectangle first.")
            return
        
        reply = QMessageBox.warning(self, "Apply to All", 
                                  "This crop operation will be applied to ALL loaded images. Do you want to continue?",
                                  QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.No:
            return

        for name, asset in self.asset_manager.images.items():
            asset.pipeline.config.setdefault("transforms", []).append({"type": "crop", "params": rect})
            asset.save_project()
        
        for name, asset in self.asset_manager.masks.items():
            asset.pipeline.config.setdefault("transforms", []).append({"type": "crop", "params": rect})
            asset.save_project()
        
        # Invalidate mask overlay cache
        self.image_handler._cached_mask_overlay = None
        
        self.cached_composite = None
        self._update_asset_list()
        self._update_mask_list()
        self._refresh_viewer()
        self.viewer_view.clear_selection()

    def _rotate_all(self):
        if not self.asset_manager.images:
            return

        line_angle = self.viewer_view.get_rotation_angle()
        if line_angle is not None:
            # line_angle is from QLineF.angle(): 0 is right, increases CCW.
            # To make the line vertical (90 or 270 deg):
            # Option 1: target 90 deg => rotation1 = 90 - line_angle
            # Option 2: target 270 deg => rotation2 = 270 - line_angle
            
            rot1 = 90 - line_angle
            rot2 = 270 - line_angle
            
            # Normalize to -180 to 180
            def normalize(a):
                while a > 180: a -= 360
                while a <= -180: a += 360
                return a
            
            rot1 = normalize(rot1)
            rot2 = normalize(rot2)
            
            # Determine which is CW and which is CCW
            # Positive is CCW in OpenCV/Qt, Negative is CW.
            # Usually users think: 
            # If rot > 0, it's CCW.
            # If rot < 0, it's CW.
            
            options = []
            if rot1 > 0: options.append(f"Counter-Clockwise ({rot1:.2f} deg)")
            else: options.append(f"Clockwise ({rot1:.2f} deg)")
            
            if rot2 > 0: options.append(f"Counter-Clockwise ({rot2:.2f} deg)")
            else: options.append(f"Clockwise ({rot2:.2f} deg)")
            
            choice, ok = QInputDialog.getItem(self, "Rotation Direction", 
                                            "Select rotation direction for perpendicular alignment:", 
                                            options, 0, False)
            if not ok:
                return
            
            angle = rot1 if choice == options[0] else rot2
            msg = f"Rotate all images by {angle:.2f} degrees?"
        else:
            angle, ok = QInputDialog.getDouble(self, "Rotate All", "Angle (degrees):", 0, -360, 360, 1)
            if not ok:
                return
            msg = f"This rotation of {angle} degrees will be applied to ALL loaded images. Do you want to continue?"

        fill_color, ok = QInputDialog.getItem(self, "Rotation Padding", 
                                        "Select padding color:", 
                                        ["Black", "White"], 0, False)
        if not ok:
            return

        reply = QMessageBox.warning(self, "Apply to All", msg,
                                  QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.No:
            return

        for name, asset in self.asset_manager.images.items():
            asset.pipeline.config.setdefault("transforms", []).append({
                "type": "rotate", 
                "angle": angle,
                "fill_color": fill_color.lower()
            })
            asset.save_project()

        for name, asset in self.asset_manager.masks.items():
            asset.pipeline.config.setdefault("transforms", []).append({
                "type": "rotate", 
                "angle": angle,
                "fill_color": fill_color.lower()
            })
            asset.save_project()

        self.cached_composite = None
        self.viewer_view.clear_rotation_line()
        self._update_asset_list()
        self._update_mask_list()
        self._refresh_viewer()

    def _export_modified_images(self):
        if not self.working_dir: return
        out_dir = QFileDialog.getExistingDirectory(self, "Select Export Directory")
        if not out_dir: return
        
        for name, asset in self.asset_manager.images.items():
            data = asset.get_rendered_data()
            if data.max() <= 1.01: 
                data = (data * 255).astype(np.uint8)
            else: 
                data = data.astype(np.uint8)
            
            # Use OpenCV to save
            if len(data.shape) == 3 and data.shape[2] == 3:
                data = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(out_dir, name), data)
        QMessageBox.information(self, "Export", "Images exported successfully.")


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

                # Move to deleted assets in JSON
                self.asset_manager.delete_mask(name)

                path = os.path.join(mask_dir, name)
                if os.path.exists(path):
                    os.remove(path)
                if name in self.visible_masks:
                    self.visible_masks.remove(name)
            self._update_mask_list()
            self.cached_composite = None
            self._refresh_viewer()

    def _save_visible_masks(self):
        if not self.visible_masks and self.preview_mask is None:
            QMessageBox.warning(self, "Save Masks", "No masks are currently visible.")
            return

        # Ensure viewer is refreshed to have the latest composite
        self._refresh_viewer()
        
        if self.cached_composite is None:
            QMessageBox.warning(self, "Save Masks", "No visible images or masks to save.")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Visible Masks", self.working_dir, "PNG Files (*.png)"
        )

        if file_path:
            if not file_path.lower().endswith('.png'):
                file_path += '.png'
            
            if self.cached_composite.save(file_path, "PNG"):
                QMessageBox.information(self, "Save Masks", f"Visible masks saved to {file_path}")
            else:
                QMessageBox.critical(self, "Save Masks", "Failed to save the image.")

    def _merge_selected_masks(self):
        selected_items = self.mask_list.selectedItems()
        if len(selected_items) < 2:
            QMessageBox.warning(self, "Merge Masks", "Please select at least two masks to merge.")
            return

        if not self.working_dir:
            return

        new_name, ok = QInputDialog.getText(self, "Merge Masks", "Enter name for the new mask:")
        if not ok or not new_name.strip():
            return

        if not new_name.lower().endswith(".png"):
            new_name += ".png"

        mask_dir = os.path.join(self.working_dir, "Cluster Masks")
        output_path = os.path.join(mask_dir, new_name)

        if os.path.exists(output_path):
            if QMessageBox.question(self, "Overwrite", f"File {new_name} already exists. Overwrite?", 
                                    QMessageBox.Yes | QMessageBox.No) == QMessageBox.No:
                return

        try:
            masks_to_merge = []
            source_masks = []
            for item in selected_items:
                mask_name = item.text()
                source_masks.append(mask_name)
                mask_asset = self.asset_manager.get_mask_by_name(mask_name)
                if mask_asset:
                    mask = mask_asset.get_rendered_data(data_only=True)
                    if mask is not None:
                        masks_to_merge.append(mask)

            if not masks_to_merge:
                return

            merged_mask = mask_refinement.merge_masks(masks_to_merge)
            cv2.imwrite(output_path, merged_mask * 255 if merged_mask.max() == 1 else merged_mask)

            # Update Project JSON
            try:
                image_list = self.asset_manager.get_image_list()
                project_name = "project_"
                if image_list:
                    parts = image_list[0].split('_')
                    if len(parts) >= 2:
                        project_name = f"{parts[0]}_{parts[1]}_"
                
                json_dir = os.path.join(self.working_dir, "JSON")
                project_json_path = os.path.join(json_dir, f"{project_name}.json")

                project_data = {}
                if os.path.exists(project_json_path):
                    with open(project_json_path, 'r') as f:
                        project_data = json.load(f)
                
                if "Masks" not in project_data:
                    project_data["Masks"] = {}
                
                project_data["Masks"][new_name] = {
                    "path": os.path.abspath(output_path),
                    "source_masks": source_masks,
                    "type": "merged"
                }

                with open(project_json_path, 'w') as f:
                    json.dump(project_data, f, indent=4)
            except Exception as json_err:
                print(f"Failed to update project JSON: {json_err}")

            self.asset_manager.scan_assets()
            self._update_mask_list()
            QMessageBox.information(self, "Merge Masks", f"Successfully merged masks into {new_name}")
        except Exception as e:
            QMessageBox.critical(self, "Merge Masks Error", f"An error occurred while merging masks: {str(e)}")

    def _asset_clicked(self, item):
        name = item.text()
        self.image_handler.toggle_visibility(name)
        self.cached_composite = None
        self._refresh_viewer()

    def get_mask_color(self, mask_name):
        # Prioritize asset config color if available
        asset = self.asset_manager.get_mask_by_name(mask_name)
        color_name = "grayscale"
        if asset:
            color_name = asset.pipeline.config.get("color", "grayscale")
        
        # Fallback to image_handler if not in asset config
        if color_name == "grayscale":
            color_name = self.image_handler.get_asset_color(mask_name)

        if color_name != "grayscale":
            rgb = self.image_handler.COLORS.get(color_name, (1, 1, 1))
            return QColor(int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))
        else:
            try:
                if mask_name.startswith("KC_"):
                    idx = int(mask_name.split('_')[1].split('.')[0])
                elif mask_name.startswith("ThresholdMask_"):
                    idx = int(mask_name.split('_')[1].split('.')[0]) + 100
                else:
                    idx = hash(mask_name)
            except:
                idx = hash(mask_name)
            color_hue = (idx * 137.5) % 360
            return QColor.fromHsvF(color_hue/360.0, 1.0, 1.0)

    def _refresh_viewer(self):
        visible_names = self.image_handler.visible_assets
        if not visible_names and not self.visible_masks and self.preview_mask is None:
            self.viewer_view.set_pixmap(None)
            self.bg_label.lower()
            return

        if self.cached_composite is None:
            # Use the new image_handler methods with caching
            img_comp = self.image_handler.render_composite(self.asset_manager)
            
            if (self.visible_masks and self.working_dir) or self.preview_mask is not None:
                # Use image_handler.render_mask_overlay for visible masks
                if self.visible_masks and self.working_dir:
                    final_comp = self.image_handler.render_mask_overlay(
                        img_comp, self.asset_manager, self.visible_masks, 
                        self.mask_opacity, self.get_mask_color
                    )
                else:
                    final_comp = img_comp

                # Preview mask is still handled here as it's transient
                if self.preview_mask is not None:
                    if final_comp is None or final_comp.isNull():
                        h, w = self.preview_mask.shape
                        composite_rgb = np.zeros((h, w, 3), dtype=np.uint8)
                    else:
                        composite_rgb = qimage2ndarray.rgb_view(final_comp).copy()

                    mask = self.preview_mask
                    if mask.shape != composite_rgb.shape[:2]:
                        mask = cv2.resize(mask, (composite_rgb.shape[1], composite_rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
                    
                    r, g, b = self.preview_color.red(), self.preview_color.green(), self.preview_color.blue()
                    mask_bool = mask.astype(bool)
                    preview_opacity = 0.7
                    overlay = composite_rgb.copy()
                    overlay[mask_bool] = [r, g, b]
                    cv2.addWeighted(overlay, preview_opacity, composite_rgb, 1 - preview_opacity, 0, composite_rgb)
                    
                    self.cached_composite = qimage2ndarray.array2qimage(composite_rgb).copy()
                else:
                    self.cached_composite = final_comp
            else:
                self.cached_composite = img_comp

        self.viewer_view.set_pixmap(self.cached_composite)
        self.bg_label.lower()

    def _create_status_bar(self):
        self.statusBar().showMessage("Ready")
