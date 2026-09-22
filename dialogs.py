from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QFormLayout, QDoubleSpinBox, QSpinBox, 
    QDialogButtonBox, QComboBox, QCheckBox, QLabel, QListWidget, 
    QPushButton, QListWidgetItem, QMessageBox, QWidget, QLineEdit,
    QTableWidget, QTableWidgetItem, QHeaderView, QHBoxLayout, QColorDialog
)
from PySide6.QtCore import Qt, Signal, QSize
from PySide6.QtGui import QColor, QPixmap, QPainter
from naming_utils import parse_image_identity

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
        elif self.filter_name == "bilateral":
            self.d_spin = QSpinBox()
            self.d_spin.setRange(1, 50)
            self.d_spin.setValue(self.params.get("d", 9))
            self.d_spin.valueChanged.connect(lambda v: self._update_param("d", v))
            form_layout.addRow("Diameter (d):", self.d_spin)

            self.sigma_color_spin = QDoubleSpinBox()
            self.sigma_color_spin.setRange(1.0, 500.0)
            self.sigma_color_spin.setValue(self.params.get("sigmaColor", 75.0))
            self.sigma_color_spin.valueChanged.connect(lambda v: self._update_param("sigmaColor", v))
            form_layout.addRow("Sigma Color:", self.sigma_color_spin)

            self.sigma_space_spin = QDoubleSpinBox()
            self.sigma_space_spin.setRange(1.0, 500.0)
            self.sigma_space_spin.setValue(self.params.get("sigmaSpace", 75.0))
            self.sigma_space_spin.valueChanged.connect(lambda v: self._update_param("sigmaSpace", v))
            form_layout.addRow("Sigma Space:", self.sigma_space_spin)

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
    def __init__(self, mask_names=None, parent=None):
        super().__init__(parent)
        self.mask_names = mask_names or []
        self.params = {
            "n_clusters": 8,
            "max_iter": 300,
            "init": "k-means++",
            "n_init": 10,
            "random_state": 0,
            "algorithm": "auto",
            "normalize": False,
            "normalize_stack": False,
            "tol": 1e-4,
            "include_coords": False,
            "x_weight": 1.0,
            "y_weight": 1.0,
            "mask_name": "None"
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
        form_layout.addRow("Normalize stack (global):", self.normalize_stack_check)

        self.mask_combo = QComboBox()
        self.mask_combo.addItem("None")
        if self.mask_names:
            self.mask_combo.addItems(self.mask_names)
        self.mask_combo.setCurrentText(self.params.get("mask_name", "None"))
        form_layout.addRow("Cluster under Mask:", self.mask_combo)

        self.include_coords_check = QCheckBox()
        self.include_coords_check.setChecked(self.params["include_coords"])
        form_layout.addRow("Include Coordinates:", self.include_coords_check)

        self.x_weight_spin = QDoubleSpinBox()
        self.x_weight_spin.setRange(0.0, 100.0)
        self.x_weight_spin.setSingleStep(0.1)
        self.x_weight_spin.setValue(self.params["x_weight"])
        form_layout.addRow("X Weight:", self.x_weight_spin)

        self.y_weight_spin = QDoubleSpinBox()
        self.y_weight_spin.setRange(0.0, 100.0)
        self.y_weight_spin.setSingleStep(0.1)
        self.y_weight_spin.setValue(self.params["y_weight"])
        form_layout.addRow("Y Weight:", self.y_weight_spin)

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
            "mask_name": self.mask_combo.currentText(),
            "include_coords": self.include_coords_check.isChecked(),
            "x_weight": self.x_weight_spin.value(),
            "y_weight": self.y_weight_spin.value()
        }

class ISODATAParameterDialog(QDialog):
    def __init__(self, mask_names=None, parent=None):
        super().__init__(parent)
        self.mask_names = mask_names or []
        self.params = {
            "n_clusters": 8,
            "max_iter": 100,
            "min_samples": 20,
            "max_std_dev": 0.1,
            "min_cluster_distance": 0.1,
            "max_merge_pairs": 2,
            "random_state": 0,
            "normalize": False,
            "normalize_stack": False,
            "include_coords": False,
            "x_weight": 1.0,
            "y_weight": 1.0,
            "mask_name": "None"
        }
        self.setup_ui()

    def setup_ui(self):
        self.setWindowTitle("ISODATA Parameters")
        layout = QVBoxLayout(self)
        form_layout = QFormLayout()

        self.n_clusters_spin = QSpinBox()
        self.n_clusters_spin.setRange(2, 100)
        self.n_clusters_spin.setValue(self.params["n_clusters"])
        form_layout.addRow("Target Clusters:", self.n_clusters_spin)

        self.max_iter_spin = QSpinBox()
        self.max_iter_spin.setRange(1, 10000)
        self.max_iter_spin.setValue(self.params["max_iter"])
        form_layout.addRow("Max Iterations:", self.max_iter_spin)

        self.min_samples_spin = QSpinBox()
        self.min_samples_spin.setRange(1, 10000)
        self.min_samples_spin.setValue(self.params["min_samples"])
        form_layout.addRow("Min Samples per Cluster:", self.min_samples_spin)

        self.max_std_dev_spin = QDoubleSpinBox()
        self.max_std_dev_spin.setRange(0.001, 100.0)
        self.max_std_dev_spin.setValue(self.params["max_std_dev"])
        form_layout.addRow("Max Std Dev for Split:", self.max_std_dev_spin)

        self.min_cluster_dist_spin = QDoubleSpinBox()
        self.min_cluster_dist_spin.setRange(0.001, 100.0)
        self.min_cluster_dist_spin.setValue(self.params["min_cluster_distance"])
        form_layout.addRow("Min Distance for Merge:", self.min_cluster_dist_spin)

        self.max_merge_pairs_spin = QSpinBox()
        self.max_merge_pairs_spin.setRange(1, 100)
        self.max_merge_pairs_spin.setValue(self.params["max_merge_pairs"])
        form_layout.addRow("Max Merge Pairs:", self.max_merge_pairs_spin)

        self.random_state_spin = QSpinBox()
        self.random_state_spin.setRange(0, 1000000)
        self.random_state_spin.setValue(self.params["random_state"])
        form_layout.addRow("Random State:", self.random_state_spin)

        self.normalize_check = QCheckBox()
        self.normalize_check.setChecked(self.params["normalize"])
        form_layout.addRow("Normalize (per image):", self.normalize_check)

        self.normalize_stack_check = QCheckBox()
        self.normalize_stack_check.setChecked(self.params["normalize_stack"])
        form_layout.addRow("Normalize stack (global):", self.normalize_stack_check)

        self.mask_combo = QComboBox()
        self.mask_combo.addItem("None")
        if self.mask_names:
            self.mask_combo.addItems(self.mask_names)
        self.mask_combo.setCurrentText(self.params.get("mask_name", "None"))
        form_layout.addRow("Cluster under Mask:", self.mask_combo)

        self.include_coords_check = QCheckBox()
        self.include_coords_check.setChecked(self.params["include_coords"])
        form_layout.addRow("Include Coordinates:", self.include_coords_check)

        self.x_weight_spin = QDoubleSpinBox()
        self.x_weight_spin.setRange(0.0, 100.0)
        self.x_weight_spin.setSingleStep(0.1)
        self.x_weight_spin.setValue(self.params["x_weight"])
        form_layout.addRow("X Weight:", self.x_weight_spin)

        self.y_weight_spin = QDoubleSpinBox()
        self.y_weight_spin.setRange(0.0, 100.0)
        self.y_weight_spin.setSingleStep(0.1)
        self.y_weight_spin.setValue(self.params["y_weight"])
        form_layout.addRow("Y Weight:", self.y_weight_spin)

        layout.addLayout(form_layout)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_params(self):
        return {
            "n_clusters": self.n_clusters_spin.value(),
            "max_iter": self.max_iter_spin.value(),
            "min_samples": self.min_samples_spin.value(),
            "max_std_dev": self.max_std_dev_spin.value(),
            "min_cluster_distance": self.min_cluster_dist_spin.value(),
            "max_merge_pairs": self.max_merge_pairs_spin.value(),
            "random_state": self.random_state_spin.value(),
            "normalize": self.normalize_check.isChecked(),
            "normalize_stack": self.normalize_stack_check.isChecked(),
            "mask_name": self.mask_combo.currentText(),
            "include_coords": self.include_coords_check.isChecked(),
            "x_weight": self.x_weight_spin.value(),
            "y_weight": self.y_weight_spin.value()
        }

class GMMParameterDialog(QDialog):
    def __init__(self, mask_names=None, parent=None):
        super().__init__(parent)
        self.mask_names = mask_names or []
        self.params = {
            "n_components": 8,
            "covariance_type": "full",
            "tol": 1e-3,
            "max_iter": 100,
            "random_state": 0,
            "normalize": False,
            "normalize_stack": False,
            "include_coords": False,
            "x_weight": 1.0,
            "y_weight": 1.0,
            "mask_name": "None"
        }
        self.setup_ui()

    def setup_ui(self):
        self.setWindowTitle("GMM Parameters")
        layout = QVBoxLayout(self)
        form_layout = QFormLayout()

        self.n_components_spin = QSpinBox()
        self.n_components_spin.setRange(2, 100)
        self.n_components_spin.setValue(self.params["n_components"])
        form_layout.addRow("Number of Components:", self.n_components_spin)

        self.cov_type_combo = QComboBox()
        self.cov_type_combo.addItems(["full", "tied", "diag", "spherical"])
        self.cov_type_combo.setCurrentText(self.params["covariance_type"])
        form_layout.addRow("Covariance Type:", self.cov_type_combo)

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
        form_layout.addRow("Normalize stack (global):", self.normalize_stack_check)

        self.mask_combo = QComboBox()
        self.mask_combo.addItem("None")
        if self.mask_names:
            self.mask_combo.addItems(self.mask_names)
        self.mask_combo.setCurrentText(self.params.get("mask_name", "None"))
        form_layout.addRow("Cluster under Mask:", self.mask_combo)

        self.include_coords_check = QCheckBox()
        self.include_coords_check.setChecked(self.params["include_coords"])
        form_layout.addRow("Include Coordinates:", self.include_coords_check)

        self.x_weight_spin = QDoubleSpinBox()
        self.x_weight_spin.setRange(0.0, 100.0)
        self.x_weight_spin.setSingleStep(0.1)
        self.x_weight_spin.setValue(self.params["x_weight"])
        form_layout.addRow("X Weight:", self.x_weight_spin)

        self.y_weight_spin = QDoubleSpinBox()
        self.y_weight_spin.setRange(0.0, 100.0)
        self.y_weight_spin.setSingleStep(0.1)
        self.y_weight_spin.setValue(self.params["y_weight"])
        form_layout.addRow("Y Weight:", self.y_weight_spin)

        layout.addLayout(form_layout)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_params(self):
        return {
            "n_components": self.n_components_spin.value(),
            "covariance_type": self.cov_type_combo.currentText(),
            "max_iter": self.max_iter_spin.value(),
            "random_state": self.random_state_spin.value(),
            "normalize": self.normalize_check.isChecked(),
            "normalize_stack": self.normalize_stack_check.isChecked(),
            "mask_name": self.mask_combo.currentText(),
            "include_coords": self.include_coords_check.isChecked(),
            "x_weight": self.x_weight_spin.value(),
            "y_weight": self.y_weight_spin.value()
        }

class ThresholdParameterDialog(QDialog):
    params_changed = Signal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Threshold Selection")
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        form_layout = QFormLayout()

        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.0, 1.0)
        self.threshold_spin.setSingleStep(0.01)
        self.threshold_spin.setValue(0.5)
        self.threshold_spin.valueChanged.connect(self._emit_params)
        form_layout.addRow("Threshold:", self.threshold_spin)

        self.normalize_check = QCheckBox()
        self.normalize_check.setChecked(True)
        self.normalize_check.stateChanged.connect(self._emit_params)
        form_layout.addRow("Normalize (0-1 range):", self.normalize_check)

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

class JointPlotDialog(QDialog):
    def __init__(self, masks, images, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Joint KDE Plot Selection")
        self.masks = masks
        self.images = images
        self.selections = []
        self.setup_ui()

    def setup_ui(self):
        self.main_layout = QVBoxLayout(self)
        
        self.sets_container = QVBoxLayout()
        self.main_layout.addLayout(self.sets_container)
        
        # Add the first set by default
        self._add_selection_set()
        
        self.add_set_btn = QPushButton("Add Another Set (Max 3)")
        self.add_set_btn.clicked.connect(self._add_selection_set)
        self.main_layout.addWidget(self.add_set_btn)

        # Filename input
        filename_layout = QFormLayout()
        self.filename_input = QLineEdit()
        self.filename_input.setPlaceholderText("Enter custom filename (optional)")
        filename_layout.addRow("Custom Filename:", self.filename_input)
        self.main_layout.addLayout(filename_layout)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._validate_and_accept)
        buttons.rejected.connect(self.reject)
        self.main_layout.addWidget(buttons)

    def _add_selection_set(self):
        if len(self.selections) >= 3:
            QMessageBox.information(self, "Limit Reached", "You can only add up to 3 sets.")
            return

        group_box = QWidget()
        group_layout = QFormLayout(group_box)
        
        set_num = len(self.selections) + 1
        group_layout.addRow(QLabel(f"<b>Set {set_num}</b>"), QLabel(""))

        mask_combo = QComboBox()
        mask_combo.addItems(self.masks)
        group_layout.addRow("Select Mask:", mask_combo)

        image1_combo = QComboBox()
        image1_combo.addItems(self.images)
        group_layout.addRow("Select Image 1 (X-axis):", image1_combo)

        image2_combo = QComboBox()
        image2_combo.addItems(self.images)
        if len(self.images) > 1:
            image2_combo.setCurrentIndex(1)
        group_layout.addRow("Select Image 2 (Y-axis):", image2_combo)

        self.sets_container.addWidget(group_box)
        self.selections.append({
            "widget": group_box,
            "mask": mask_combo,
            "image1": image1_combo,
            "image2": image2_combo
        })
        
        if len(self.selections) >= 3:
            self.add_set_btn.setEnabled(False)

    def _validate_and_accept(self):
        for i, sel in enumerate(self.selections):
            if sel["image1"].currentText() == sel["image2"].currentText():
                QMessageBox.warning(self, "Invalid Selection", f"Set {i+1}: Please select two different images.")
                return
        self.accept()

    def get_selections(self):
        results = []
        for sel in self.selections:
            results.append({
                "mask": sel["mask"].currentText(),
                "image1": sel["image1"].currentText(),
                "image2": sel["image2"].currentText()
            })
        return results, self.filename_input.text().strip()

class RefineMaskDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Refine Mask Parameters")
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        form_layout = QFormLayout()

        self.min_area_spin = QSpinBox()
        self.min_area_spin.setRange(0, 1000000)
        self.min_area_spin.setValue(40)
        form_layout.addRow("Minimum Area (pixels):", self.min_area_spin)

        self.min_circularity_spin = QDoubleSpinBox()
        self.min_circularity_spin.setRange(0.0, 1.0)
        self.min_circularity_spin.setSingleStep(0.05)
        self.min_circularity_spin.setValue(0.0)
        form_layout.addRow("Minimum Circularity:", self.min_circularity_spin)

        self.max_eccentricity_spin = QDoubleSpinBox()
        self.max_eccentricity_spin.setRange(0.0, 1.0)
        self.max_eccentricity_spin.setSingleStep(0.05)
        self.max_eccentricity_spin.setValue(0.95)
        form_layout.addRow("Maximum Eccentricity:", self.max_eccentricity_spin)

        self.min_solidity_spin = QDoubleSpinBox()
        self.min_solidity_spin.setRange(0.0, 1.0)
        self.min_solidity_spin.setSingleStep(0.05)
        self.min_solidity_spin.setValue(0.35)
        form_layout.addRow("Minimum Solidity:", self.min_solidity_spin)

        self.min_extent_spin = QDoubleSpinBox()
        self.min_extent_spin.setRange(0.0, 1.0)
        self.min_extent_spin.setSingleStep(0.05)
        self.min_extent_spin.setValue(0.0)
        form_layout.addRow("Minimum Extent:", self.min_extent_spin)

        self.euler_number_check = QCheckBox("Filter by Euler Number")
        self.euler_number_spin = QSpinBox()
        self.euler_number_spin.setRange(-100, 100)
        self.euler_number_spin.setValue(1)
        self.euler_number_spin.setEnabled(False)
        self.euler_number_check.toggled.connect(self.euler_number_spin.setEnabled)
        
        euler_layout = QHBoxLayout()
        euler_layout.addWidget(self.euler_number_check)
        euler_layout.addWidget(self.euler_number_spin)
        form_layout.addRow("Euler Number:", euler_layout)

        layout.addLayout(form_layout)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_params(self):
        return {
            "min_area": self.min_area_spin.value(),
            "min_circularity": self.min_circularity_spin.value(),
            "max_eccentricity": self.max_eccentricity_spin.value(),
            "min_solidity": self.min_solidity_spin.value(),
            "min_extent": self.min_extent_spin.value(),
            "euler_number": self.euler_number_spin.value() if self.euler_number_check.isChecked() else None
        }

class MaskPropertiesDialog(QDialog):
    def __init__(self, properties, mask_name, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Properties: {mask_name}")
        self.properties = properties
        self.setup_ui()

    def setup_ui(self):
        self.resize(800, 400)
        layout = QVBoxLayout(self)

        self.table = QTableWidget()
        self.table.setColumnCount(7)
        self.table.setHorizontalHeaderLabels([
            "Label", "Area (px)", "Eccentricity", "Circularity", 
            "Solidity", "Extent", "Euler Number"
        ])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        
        self.table.setRowCount(len(self.properties))
        for i, prop in enumerate(self.properties):
            self.table.setItem(i, 0, QTableWidgetItem(str(prop["Label"])))
            self.table.setItem(i, 1, QTableWidgetItem(str(prop["Area"])))
            self.table.setItem(i, 2, QTableWidgetItem(str(prop["Eccentricity"])))
            self.table.setItem(i, 3, QTableWidgetItem(str(prop["Circularity"])))
            self.table.setItem(i, 4, QTableWidgetItem(str(prop.get("Solidity", "N/A"))))
            self.table.setItem(i, 5, QTableWidgetItem(str(prop.get("Extent", "N/A"))))
            self.table.setItem(i, 6, QTableWidgetItem(str(prop.get("Euler Number", "N/A"))))

        layout.addWidget(self.table)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)

class ProbeColorRuleDialog(QDialog):
    def __init__(self, probe_names, initial_colors=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Set Color Rules for Probes")
        self.resize(400, 500)
        self.probe_names = sorted(list(set(probe_names)))
        self.colors = initial_colors.copy() if initial_colors else {}
        self.color_buttons = {}
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        from PySide6.QtWidgets import QScrollArea
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        
        for probe in self.probe_names:
            h_layout = QHBoxLayout()
            label = QLabel(probe)
            
            # Create a color button
            btn = QPushButton()
            btn.setFixedWidth(50)
            color = self.colors.get(probe, "#ffffff")
            self._update_button_color(btn, color)
            btn.clicked.connect(lambda checked=False, p=probe, b=btn: self._pick_color(p, b))
            
            h_layout.addWidget(label)
            h_layout.addStretch()
            h_layout.addWidget(btn)
            scroll_layout.addLayout(h_layout)
            self.color_buttons[probe] = btn
            
        scroll.setWidget(scroll_content)
        layout.addWidget(scroll)
        
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _update_button_color(self, button, color_hex):
        pixmap = QPixmap(30, 20)
        pixmap.fill(QColor(color_hex))
        button.setIcon(pixmap)
        button.setIconSize(QSize(30, 20))

    def _pick_color(self, probe, button):
        initial_color = QColor(self.colors.get(probe, "#ffffff"))
        color = QColorDialog.getColor(initial_color, self, f"Select Color for {probe}")
        if color.isValid():
            hex_color = color.name()
            self.colors[probe] = hex_color
            self._update_button_color(button, hex_color)

    def get_colors(self):
        return self.colors

class MetadataAssignmentDialog(QDialog):
    def __init__(self, file_list, parent=None):
        super().__init__(parent)
        self.file_list = file_list
        self.results = {}
        self.setup_ui()
        self.auto_assign_all()

    def setup_ui(self):
        self.setWindowTitle("Assign File Metadata")
        self.resize(1000, 600)
        layout = QVBoxLayout(self)

        instruction = QLabel("Review and specify metadata for the files in the working directory.")
        layout.addWidget(instruction)

        self.table = QTableWidget(len(self.file_list), 7)
        self.table.setHorizontalHeaderLabels([
            "Original Filename", "Sample", "Slide ##", "Owner", 
            "Objective", "Well Position", "Probe"
        ])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)

        for i, filename in enumerate(self.file_list):
            self.table.setItem(i, 0, QTableWidgetItem(filename))
            self.table.item(i, 0).setFlags(self.table.item(i, 0).flags() & ~Qt.ItemIsEditable)
            
            for j in range(1, 7):
                self.table.setItem(i, j, QTableWidgetItem(""))

        layout.addWidget(self.table)

        btn_layout = QHBoxLayout()
        self.auto_btn = QPushButton("Auto-Assign All")
        self.auto_btn.clicked.connect(self.auto_assign_all)
        btn_layout.addWidget(self.auto_btn)
        
        btn_layout.addStretch()
        
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.validate_and_accept)
        buttons.rejected.connect(self.reject)
        btn_layout.addWidget(buttons)
        
        layout.addLayout(btn_layout)

    def auto_assign_all(self):
        for i in range(self.table.rowCount()):
            filename = self.table.item(i, 0).text()
            identity = parse_image_identity(filename)
            
            if identity.sample: self.table.setItem(i, 1, QTableWidgetItem(identity.sample))
            if identity.slide: self.table.setItem(i, 2, QTableWidgetItem(identity.slide))
            if identity.owner: self.table.setItem(i, 3, QTableWidgetItem(identity.owner))
            if identity.objective: self.table.setItem(i, 4, QTableWidgetItem(identity.objective))
            if identity.well_position: self.table.setItem(i, 5, QTableWidgetItem(identity.well_position))
            if identity.probe: self.table.setItem(i, 6, QTableWidgetItem(identity.probe))

    def validate_and_accept(self):
        self.results = {}
        for i in range(self.table.rowCount()):
            filename = self.table.item(i, 0).text()
            metadata = []
            for j in range(1, 7):
                val = self.table.item(i, j).text().strip()
                if not val:
                    QMessageBox.warning(self, "Missing Data", f"Please fill all fields for {filename}")
                    return
                metadata.append(val)
            
            # Simple validation to match naming convention expectations
            # Sample: Numbers
            if not metadata[0].isdigit():
                QMessageBox.warning(self, "Invalid Data", f"Sample must be numeric for {filename}")
                return
            # Slide ##: Alpha-numeric
            if not metadata[1].isalnum():
                QMessageBox.warning(self, "Invalid Data", f"Slide ## must be alpha-numeric for {filename}")
                return
            # Owner: Letters
            if not metadata[2].isalpha():
                QMessageBox.warning(self, "Invalid Data", f"Owner Initials must be letters for {filename}")
                return
            # Objective: Number + x/X
            if not (metadata[3][:-1].isdigit() and metadata[3][-1].lower() == 'x'):
                QMessageBox.warning(self, "Invalid Data", f"Objective must be number + 'x' for {filename}")
                return
            # Well Position: Number 1-12
            if not (metadata[4].isdigit() and 1 <= int(metadata[4]) <= 12):
                QMessageBox.warning(self, "Invalid Data", f"Well Position must be 1-12 for {filename}")
                return
            # Probe: Alpha-numeric
            if not metadata[5].isalnum():
                QMessageBox.warning(self, "Invalid Data", f"Probe must be alpha-numeric for {filename}")
                return

            self.results[filename] = metadata
            
        self.accept()

    def get_results(self):
        return self.results

