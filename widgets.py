from PySide6.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QGraphicsRectItem, QGraphicsLineItem, QFrame
from PySide6.QtGui import QPixmap, QPalette, QPen, QColor, QImage, QBrush, QWheelEvent
from PySide6.QtCore import Qt, QPointF, QRectF, QSize, QLineF

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
        self.rotation_line_item = None
        self.start_scene_pos = None
        self.moving_selection = False
        self.drawing_line = False
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
        self.clear_rotation_line()

    def clear_selection(self):
        if self.selection_rect_item:
            self.scene.removeItem(self.selection_rect_item)
            self.selection_rect_item = None
        self.start_scene_pos = None

    def clear_rotation_line(self):
        if self.rotation_line_item:
            self.scene.removeItem(self.rotation_line_item)
            self.rotation_line_item = None

    def mousePressEvent(self, event):
        modifiers = event.modifiers()
        if event.button() == Qt.LeftButton:
            scene_pos = self.mapToScene(event.pos())
            
            if modifiers == (Qt.ControlModifier | Qt.AltModifier):
                self.drawing_line = True
                self.start_scene_pos = scene_pos
                self.clear_rotation_line()
                self.rotation_line_item = QGraphicsLineItem()
                pen = QPen(QColor(255, 0, 0), 2)
                pen.setCosmetic(True)
                self.rotation_line_item.setPen(pen)
                self.scene.addItem(self.rotation_line_item)
                self.rotation_line_item.setLine(QLineF(self.start_scene_pos, self.start_scene_pos))
                return

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
        if self.start_scene_pos is not None:
            current_scene_pos = self.mapToScene(event.pos())
            img_rect = self.pixmap_item.boundingRect()

            if self.drawing_line and self.rotation_line_item:
                self.rotation_line_item.setLine(QLineF(self.start_scene_pos, current_scene_pos))
            elif self.selection_rect_item:
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
            self.drawing_line = False
        super().mouseReleaseEvent(event)

    def get_selection_rect(self):
        if self.selection_rect_item:
            rect = self.selection_rect_item.rect()
            return (int(rect.x()), int(rect.y()), int(rect.width()), int(rect.height()))
        return None

    def get_rotation_angle(self):
        if self.rotation_line_item:
            line = self.rotation_line_item.line()
            if line.length() > 0:
                # QLineF angle is in degrees, 0 is at 3 o'clock, counter-clockwise
                # We want perpendicular to this line.
                # If we want the line to become vertical (90 or 270 deg),
                # the rotation angle should be such that line_angle + rotation = 90 (or -90)
                return line.angle()
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
