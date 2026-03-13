from PySide6.QtWidgets import (
    QMainWindow, QMenu, QMenuBar, QFileDialog, QListWidget, QListWidgetItem, 
    QSplitter, QWidget, QVBoxLayout, QLabel, QScrollArea, QMdiArea, QMdiSubWindow, QDockWidget
)
from PySide6.QtGui import QAction, QPixmap, QIcon, QPainter
from PySide6.QtCore import Qt, QSize, QPoint
import os
from assets import AssetManager
from image_handler import ImageDisplayHandler

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Clustering App")
        self.resize(1500, 1500)

        self.asset_manager = AssetManager()
        self.image_handler = ImageDisplayHandler()
        self.working_dir = None

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
        self.image_list = QListWidget()
        self.image_list.setIconSize(QSize(100, 100))
        self.image_list.setSelectionMode(QListWidget.MultiSelection)
        self.image_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.image_list.customContextMenuRequested.connect(self._show_image_context_menu)
        self.image_list.itemClicked.connect(self._asset_clicked)
        dock_layout.addWidget(QLabel("Images:"))
        dock_layout.addWidget(self.image_list)

        # Mask list
        self.mask_list = QListWidget()
        self.mask_list.setIconSize(QSize(100, 100))
        self.mask_list.setSelectionMode(QListWidget.SingleSelection)
        # self.mask_list.itemClicked.connect(self._mask_clicked) # TODO: Mask interaction
        dock_layout.addWidget(QLabel("Masks:"))
        dock_layout.addWidget(self.mask_list)

        dock.setWidget(dock_container)

        # Create the image viewer in a sub-window
        self.viewer_subwindow = QMdiSubWindow()
        self.viewer_subwindow.setWindowTitle("Image Viewer")
        
        self.viewer_container = QScrollArea()
        self.viewer_label = QLabel()
        self.viewer_label.setAlignment(Qt.AlignCenter)
        self.viewer_label.setText("No images selected")
        self.viewer_container.setWidget(self.viewer_label)
        self.viewer_container.setWidgetResizable(True)
        
        self.viewer_subwindow.setWidget(self.viewer_container)
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
        # Placeholder actions to be defined later
        cluster_menu.addAction("Placeholder")

        # "Tools" dropdown button
        tools_menu = menu_bar.addMenu("Tools")
        # Placeholder actions to be defined later
        tools_menu.addAction("Placeholder")

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
        for name in sorted(self.asset_manager.masks.keys()):
            mask_asset = self.asset_manager.masks[name]
            
            # Create thumbnail
            qimg = mask_asset.to_qimage()
            pixmap = QPixmap.fromImage(qimg)
            thumbnail = pixmap.scaled(100, 100, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            
            item = QListWidgetItem(name)
            item.setIcon(QIcon(thumbnail))
            self.mask_list.addItem(item)

    def _show_image_context_menu(self, position: QPoint):
        item = self.image_list.itemAt(position)
        if not item:
            return

        name = item.text()
        menu = QMenu()
        
        color_menu = menu.addMenu("Change Color")
        for color_name in self.image_handler.COLORS.keys():
            action = QAction(color_name.capitalize(), self)
            action.triggered.connect(lambda checked=False, n=name, c=color_name: self._change_color(n, c))
            color_menu.addAction(action)
            
        menu.exec(self.image_list.mapToGlobal(position))

    def _change_color(self, name, color_name):
        self.image_handler.set_asset_color(name, color_name)
        # Update tooltip
        for i in range(self.image_list.count()):
            item = self.image_list.item(i)
            if item.text() == name:
                item.setToolTip(f"Color: {color_name}")
                break
        self._refresh_viewer()

    def _asset_clicked(self, item):
        name = item.text()
        self.image_handler.toggle_visibility(name)
        
        # Visually reflect selection state
        item.setSelected(self.image_handler.is_visible(name))
        
        self._refresh_viewer()

    def _refresh_viewer(self):
        composite_qimg = self.image_handler.render_composite(self.asset_manager)
        if composite_qimg:
            pixmap = QPixmap.fromImage(composite_qimg)
            self.viewer_label.setPixmap(pixmap)
        else:
            self.viewer_label.clear()
            self.viewer_label.setText("No images selected")

    def _create_status_bar(self):
        self.statusBar().showMessage("Ready")
