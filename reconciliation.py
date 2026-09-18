import os
import json
from PySide6.QtWidgets import QDialog, QVBoxLayout, QLabel, QCheckBox, QPushButton, QScrollArea, QWidget, QHBoxLayout
from PySide6.QtCore import Qt

class ReconciliationDialog(QDialog):
    def __init__(self, missing_items, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Reconcile Project Files")
        self.setMinimumSize(500, 400)
        
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("The following items were listed in the project JSON but are missing from disk."))
        layout.addWidget(QLabel("Select the items you wish to KEEP in the project JSON (unchecked items will be removed):"))
        
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll_content = QWidget()
        self.scroll_layout = QVBoxLayout(self.scroll_content)
        
        self.checkboxes = {}
        
        has_items = False
        if missing_items.get("Images"):
            has_items = True
            self.scroll_layout.addWidget(QLabel("<b>Images:</b>"))
            for item in missing_items["Images"]:
                cb = QCheckBox(item)
                cb.setChecked(True)
                self.scroll_layout.addWidget(cb)
                self.checkboxes[("Image", item)] = cb
                
        if missing_items.get("Masks"):
            has_items = True
            self.scroll_layout.addWidget(QLabel("<b>Masks:</b>"))
            for item in missing_items["Masks"]:
                cb = QCheckBox(item)
                cb.setChecked(True)
                self.scroll_layout.addWidget(cb)
                self.checkboxes[("Mask", item)] = cb

        if missing_items.get("Phenotypes"):
            has_items = True
            self.scroll_layout.addWidget(QLabel("<b>Mask Phenotypes:</b>"))
            for item in missing_items["Phenotypes"]:
                cb = QCheckBox(item)
                cb.setChecked(True)
                self.scroll_layout.addWidget(cb)
                self.checkboxes[("Phenotype", item)] = cb
        
        if not has_items:
            self.scroll_layout.addWidget(QLabel("No missing items found."))
            
        self.scroll.setWidget(self.scroll_content)
        layout.addWidget(self.scroll)
        
        btn_layout = QHBoxLayout()
        ok_btn = QPushButton("OK")
        ok_btn.clicked.connect(self.accept)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addStretch()
        btn_layout.addWidget(ok_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)

    def get_selected_to_keep(self):
        """Returns a list of (type, name) tuples that should be kept."""
        return [key for key, cb in self.checkboxes.items() if cb.isChecked()]

def reconcile_project_with_disk(asset_manager, parent=None):
    if not asset_manager.working_dir:
        return

    project_json_path = asset_manager.get_project_json_path()
    if not project_json_path or not os.path.exists(project_json_path):
        return

    try:
        with open(project_json_path, 'r') as f:
            project_data = json.load(f)
    except Exception as e:
        print(f"Error reading project JSON: {e}")
        return

    missing_items = {"Images": [], "Masks": [], "Phenotypes": []}
    
    # Check Images
    image_ids = project_data.get("Image IDs", [])
    for img_id in image_ids:
        # Check by filename in working directory
        if not os.path.exists(os.path.join(asset_manager.working_dir, img_id)):
            missing_items["Images"].append(img_id)
            
    # Check Masks
    masks_data = project_data.get("Masks", {})
    mask_dir = os.path.join(asset_manager.working_dir, "Cluster Masks")
    for mask_name in masks_data.keys():
        if not os.path.exists(os.path.join(mask_dir, mask_name)):
            missing_items["Masks"].append(mask_name)

    # Check Mask Phenotypes
    phenotypes_data = project_data.get("Mask Phenotypes", [])
    for entry in phenotypes_data:
        csv_path = entry.get("csv_path")
        if csv_path and not os.path.exists(csv_path):
            missing_items["Phenotypes"].append(csv_path)

    if not missing_items["Images"] and not missing_items["Masks"] and not missing_items["Phenotypes"]:
        return

    dialog = ReconciliationDialog(missing_items, parent)
    if dialog.exec() == QDialog.Accepted:
        selected_to_keep = dialog.get_selected_to_keep()
        
        # We want to remove items that are missing AND were NOT checked (to keep)
        to_remove_images = [img for img in missing_items["Images"] if ("Image", img) not in selected_to_keep]
        to_remove_masks = [mask for mask in missing_items["Masks"] if ("Mask", mask) not in selected_to_keep]
        to_remove_phenotypes = [p for p in missing_items["Phenotypes"] if ("Phenotype", p) not in selected_to_keep]
        
        if not to_remove_images and not to_remove_masks and not to_remove_phenotypes:
            return

        # Perform removal
        for img_id in to_remove_images:
            if img_id in project_data.get("Image IDs", []):
                project_data["Image IDs"].remove(img_id)
            if img_id in project_data.get("Image Paths", {}):
                del project_data["Image Paths"][img_id]
            if img_id in project_data.get("Image JSON Paths", {}):
                del project_data["Image JSON Paths"][img_id]
                
        for mask_name in to_remove_masks:
            if mask_name in project_data.get("Masks", {}):
                del project_data["Masks"][mask_name]

        if to_remove_phenotypes:
            project_data["Mask Phenotypes"] = [
                entry for entry in project_data.get("Mask Phenotypes", [])
                if entry.get("csv_path") not in to_remove_phenotypes
            ]
                
        # Save back to project JSON
        try:
            with open(project_json_path, 'w') as f:
                json.dump(project_data, f, indent=4)
        except Exception as e:
            print(f"Error saving project JSON: {e}")
            
        # Also need to notify AssetManager to rescans if needed, 
        # but AssetManager.scan_assets() usually looks at disk anyway.
