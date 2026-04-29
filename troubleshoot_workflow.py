import os
os.environ['QT_API'] = 'pyside6'
import sys
import datetime
import numpy as np
import traceback

# Ensure we can import modules from the project root
sys.path.append(os.path.abspath("C:/Users/ERK220/PycharmProjects/CMPp6"))

# Redirect output for logging
class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

log_file = "troubleshoot_log.txt"
sys.stdout = Logger(log_file)
sys.stderr = sys.stdout

def log_step(step_name):
    print(f"\n{'='*20} STEP: {step_name} {'='*20}")
    print(f"Time: {datetime.datetime.now()}")

def main():
    try:
        from assets import AssetManager
        import image_stacker
        import clustering
        import measure_utilities
        import histogram_plots

        target_dir = r"C:\Users\ERK220\PycharmProjects\CMPp6\Test Folder\13907_R1"
        
        # 1. Navigate to home folder (Setup AssetManager)
        log_step("Navigating to home folder and scanning assets")
        am = AssetManager()
        am.set_working_dir(target_dir)
        am.scan_assets()
        image_names = am.get_image_list()
        print(f"Working Directory: {target_dir}")
        print(f"Images found: {image_names}")
        
        if not image_names:
            print("ERROR: No images found in the target directory.")
            return

        # 2. Invert the images
        log_step("Inverting all images")
        for name in image_names:
            asset = am.get_image_by_name(name)
            asset.load_project()
            # Toggle invert
            asset.pipeline.config["invert"] = True
            asset.save_project()
            print(f"Inverted image: {name}")

        # 3. Runs a GMM clustering with number = 4
        log_step("Running GMM clustering (n_components=4)")
        stack = image_stacker.load_and_stack_images(am, image_names)
        if stack is None:
            print("ERROR: Failed to create image stack.")
            return
        
        stack = stack.astype(np.float32)
        params = {
            "n_components": 4,
            "covariance_type": "full",
            "tol": 1e-3,
            "max_iter": 100,
            "random_state": 0,
            "normalize": False,
            "normalize_stack": False
        }
        
        mask_root_name = "troubleshoot_gmm"
        # GMM clustering
        result_mask = clustering.gaussian_mixture_clustering(
            stack, 
            mask=None, 
            include_coords=False, 
            x_weight=1.0, 
            y_weight=1.0,
            **params
        )
        print(f"GMM clustering completed. Result mask shape: {result_mask.shape}")
        
        # Save the full cluster mask
        mask_dir = os.path.join(target_dir, "Cluster Masks")
        os.makedirs(mask_dir, exist_ok=True)
        full_mask_path = os.path.join(mask_dir, f"{mask_root_name}.npy")
        np.save(full_mask_path, result_mask)
        print(f"Full cluster mask saved to: {full_mask_path}")

        # 4. Use each of the produced masks to create histograms for each image
        log_step("Creating histograms for each cluster mask")
        
        individual_masks = clustering.get_individual_masks(result_mask, 4)
        graph_dir = os.path.join(target_dir, "Graphs")
        os.makedirs(graph_dir, exist_ok=True)
        
        for i, mask in enumerate(individual_masks):
            cluster_id = i + 1
            mask_name = f"{mask_root_name}_cluster_{cluster_id}.npy"
            mask_path = os.path.join(mask_dir, mask_name)
            np.save(mask_path, mask)
            print(f"\nProcessing Cluster {cluster_id}")
            
            measurements = measure_utilities.calculate_mask_measurements(am, image_names, mask_path)
            if measurements:
                hist_files = histogram_plots.create_histograms(measurements, mask_name, graph_dir)
                print(f"Generated {len(hist_files)} individual histograms.")
            else:
                print(f"WARNING: Failed to calculate measurements for cluster {cluster_id}")

        log_step("Workflow completed successfully")

    except Exception as e:
        print("\n" + "!"*20 + " ERROR OCCURRED " + "!"*20)
        traceback.print_exc()
        print("!"*56)

if __name__ == "__main__":
    main()
