import os
import numpy as np
import cv2
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
    QSpinBox, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, 
    QFileDialog, QMessageBox, QComboBox, QToolBar, QStatusBar,
    QGraphicsEllipseItem
)
from PySide6.QtGui import (
    QPixmap, QImage, QPainter, QPen, QColor, QBrush, QAction, QIcon, QShortcut, QKeySequence,
    QWheelEvent
)
from PySide6.QtCore import Qt, QPoint, QRectF, QSize, Signal, QPointF
import qimage2ndarray

class PaintView(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setRenderHint(QPainter.Antialiasing)
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        
        self.image_item = QGraphicsPixmapItem()
        self.mask_item = QGraphicsPixmapItem()
        self.mask_item.setOpacity(0.5)
        
        self.scene.addItem(self.image_item)
        self.scene.addItem(self.mask_item)
        
        # Cursor for brush size preview
        self.cursor_item = QGraphicsEllipseItem()
        self.cursor_item.setPen(QPen(QColor(255, 255, 0), 1))
        self.scene.addItem(self.cursor_item)
        self.cursor_item.setZValue(10)
        
        self.painting = False
        self.last_point = QPoint()
        self.brush_size = 10
        self.mode = "paint"  # "paint", "erase", "fill"
        self.mask_qimage = None
        
    def set_images(self, image_qimage, mask_qimage):
        self.image_item.setPixmap(QPixmap.fromImage(image_qimage))
        self.mask_qimage = mask_qimage
        self.update_mask_display()
        self.scene.setSceneRect(QRectF(image_qimage.rect()))

    def update_mask_display(self):
        if self.mask_qimage:
            self.mask_item.setPixmap(QPixmap.fromImage(self.mask_qimage))

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.painting = True
            self.last_point = self.mapToScene(event.pos()).toPoint()
            if self.mode == "fill":
                self.flood_fill(self.last_point)
            else:
                self.draw_on_mask(self.last_point, self.last_point)
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        scene_pos = self.mapToScene(event.pos())
        self.update_cursor(scene_pos)
        
        if self.painting and self.mode != "fill":
            current_point = scene_pos.toPoint()
            self.draw_on_mask(self.last_point, current_point)
            self.last_point = current_point
        else:
            super().mouseMoveEvent(event)

    def update_cursor(self, scene_pos):
        r = self.brush_size / 2
        if isinstance(scene_pos, QPointF):
            self.cursor_item.setRect(scene_pos.x() - r, scene_pos.y() - r, self.brush_size, self.brush_size)
        else:
            # Handle QPoint or other types if necessary
            self.cursor_item.setRect(scene_pos.x() - r, scene_pos.y() - r, self.brush_size, self.brush_size)
            
        if self.mode == "fill":
            self.cursor_item.hide()
        else:
            self.cursor_item.show()

    def enterEvent(self, event):
        self.cursor_item.show()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self.cursor_item.hide()
        super().leaveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.painting = False
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event: QWheelEvent):
        if event.modifiers() == Qt.ControlModifier:
            factor = 1.15
            if event.angleDelta().y() < 0:
                factor = 1.0 / factor
            self.scale(factor, factor)
        elif event.modifiers() == Qt.NoModifier:
            # Change brush size with scroll wheel
            delta = 1 if event.angleDelta().y() > 0 else -1
            new_size = max(1, min(100, self.brush_size + delta))
            if new_size != self.brush_size:
                # We need a way to update the spin box in the dialog
                # Let's emit a signal or call a parent method
                if hasattr(self.parent(), 'brush_spin'):
                    self.parent().brush_spin.setValue(new_size)
                else:
                    self.brush_size = new_size
                    self.update_cursor(self.mapToScene(event.pos()))
        else:
            super().wheelEvent(event)

    def draw_on_mask(self, start_point, end_point):
        if not self.mask_qimage:
            return
        
        painter = QPainter(self.mask_qimage)
        if self.mode == "paint":
            painter.setPen(QPen(QColor(255, 255, 255), self.brush_size, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        elif self.mode == "erase":
            painter.setCompositionMode(QPainter.CompositionMode_Source)
            painter.setPen(QPen(QColor(0, 0, 0), self.brush_size, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        
        painter.drawLine(start_point, end_point)
        painter.end()
        self.update_mask_display()

    def flood_fill(self, point):
        if not self.mask_qimage:
            return
        
        # Convert to numpy for easier flood fill
        mask_array = qimage2ndarray.rgb_view(self.mask_qimage)
        # Assuming binary mask (white on transparent/black)
        # OpenCV floodFill needs a single channel image or 3 channel
        # Let's use qimage2ndarray to get a 0-255 grayscale
        gray_mask = cv2.cvtColor(mask_array, cv2.COLOR_RGB2GRAY)
        
        h, w = gray_mask.shape
        seed_point = (point.x(), point.y())
        if seed_point[0] < 0 or seed_point[0] >= w or seed_point[1] < 0 or seed_point[1] >= h:
            return
            
        # OpenCV floodFill needs a mask of size (h+2, w+2)
        ff_mask = np.zeros((h + 2, w + 2), np.uint8)
        new_val = 255 if self.mode != "erase" else 0 # actually the mode fill will only be called when mode is fill
        # But we might want to fill with black if we want to "erase" a region
        # For now, flood fill always fills with white
        cv2.floodFill(gray_mask, ff_mask, seed_point, 255)
        
        # Convert back
        new_mask = cv2.cvtColor(gray_mask, cv2.COLOR_GRAY2RGB)
        # We need to preserve the original format if possible, but RGB32 is fine
        self.mask_qimage = QImage(new_mask.data, w, h, w * 3, QImage.Format_RGB888).copy()
        self.mask_qimage = self.mask_qimage.convertToFormat(QImage.Format_RGB32)
        self.update_mask_display()

class PaintDialog(QDialog):
    def __init__(self, image_asset, mask_data=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Paint Mask")
        self.resize(1000, 800)
        
        self.image_asset = image_asset
        self.image_qimage = image_asset.to_qimage(for_display=False)
        
        if mask_data is not None:
            # mask_data is expected to be a numpy array (binary mask)
            # Ensure it is uint8 for OpenCV
            mask_uint8 = (mask_data * 255).astype(np.uint8) if mask_data.max() <= 1.0 else mask_data.astype(np.uint8)
            h, w = mask_uint8.shape[:2]
            rgb_mask = cv2.cvtColor(mask_uint8, cv2.COLOR_GRAY2RGB)
            self.mask_qimage = QImage(rgb_mask.data, w, h, w * 3, QImage.Format_RGB888).copy()
            self.mask_qimage = self.mask_qimage.convertToFormat(QImage.Format_RGB32)
        else:
            # Create a blank mask
            self.mask_qimage = QImage(self.image_qimage.size(), QImage.Format_RGB32)
            self.mask_qimage.fill(Qt.black)
        
        self.setup_ui()
        self.view.set_images(self.image_qimage, self.mask_qimage)

    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Toolbar
        toolbar = QHBoxLayout()
        
        self.paint_btn = QPushButton("Paint")
        self.paint_btn.setCheckable(True)
        self.paint_btn.setChecked(True)
        self.paint_btn.clicked.connect(lambda: self.set_mode("paint"))
        toolbar.addWidget(self.paint_btn)
        
        self.erase_btn = QPushButton("Erase")
        self.erase_btn.setCheckable(True)
        self.erase_btn.clicked.connect(lambda: self.set_mode("erase"))
        toolbar.addWidget(self.erase_btn)
        
        self.fill_btn = QPushButton("Fill")
        self.fill_btn.setCheckable(True)
        self.fill_btn.clicked.connect(lambda: self.set_mode("fill"))
        toolbar.addWidget(self.fill_btn)
        
        toolbar.addWidget(QLabel("Brush Size:"))
        self.brush_spin = QSpinBox()
        self.brush_spin.setRange(1, 100)
        self.brush_spin.setValue(10)
        self.brush_spin.valueChanged.connect(self.update_brush_size)
        toolbar.addWidget(self.brush_spin)
        
        toolbar.addStretch()
        
        self.save_btn = QPushButton("Save Mask")
        self.save_btn.clicked.connect(self.accept)
        toolbar.addWidget(self.save_btn)
        
        layout.addLayout(toolbar)
        
        # View
        self.view = PaintView()
        layout.addWidget(self.view)
        
        # Status bar
        self.status_bar = QLabel("Mode: Paint | Brush Size: 10")
        layout.addWidget(self.status_bar)

    def set_mode(self, mode):
        self.view.mode = mode
        self.paint_btn.setChecked(mode == "paint")
        self.erase_btn.setChecked(mode == "erase")
        self.fill_btn.setChecked(mode == "fill")
        self.update_status()

    def update_brush_size(self, size):
        self.view.brush_size = size
        # Update cursor size immediately if it's visible
        # Use the stored mouse position or global mouse pos if possible
        # For simplicity, just update it where it is
        self.view.update_cursor(self.view.cursor_item.rect().center())
        self.update_status()

    def update_status(self):
        self.status_bar.setText(f"Mode: {self.view.mode.capitalize()} | Brush Size: {self.view.brush_size}")

    def get_mask(self):
        # Convert QImage to numpy binary mask
        mask_array = qimage2ndarray.rgb_view(self.view.mask_qimage)
        gray = cv2.cvtColor(mask_array, cv2.COLOR_RGB2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        return binary
