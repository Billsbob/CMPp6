import numpy as np
import os
from assets import AssetManager

def load_and_stack_images(asset_manager: AssetManager, image_names=None):
    """
    Takes the image names (or all if None) from the asset_manager,
    applies their individual transformations (crop, invert, etc.) 
    as defined in their .json sidecar files, and stacks them into a numpy array.
    
    Returns:
        np.ndarray: A stack of processed image data with shape (N, H, W)
    """
    if image_names is None:
        image_names = asset_manager.get_image_list()
    
    if not image_names:
        print("No images to stack.")
        return None
    
    processed_images = []
    target_shape = None
    target_dtype = None
    target_channels = None
    
    for name in image_names:
        asset = asset_manager.get_image_by_name(name)
        if not asset:
            print(f"Warning: Asset {name} not found.")
            continue
        
        # Ensure project/json is loaded
        asset.load_project()
        
        raw_data = asset.data
        if raw_data is None:
            print(f"Warning: Could not load raw data for {name}.")
            continue
            
        # get_rendered_data with data_only=True applies Crop, Rotate, Invert, and Filters
        # but excludes display-only transformations like contrast stretch and normalization.
        data = asset.get_rendered_data(data_only=True)
        
        if data is None:
            print(f"Warning: Could not process data for {name}.")
            continue
            
        # Bit depth (dtype) and raw color (channels) check
        # We check the RAW data for these properties.
        current_dtype = raw_data.dtype
        current_channels = 1 if len(raw_data.shape) == 2 else raw_data.shape[2]
        
        if target_dtype is None:
            target_dtype = current_dtype
            target_channels = current_channels
        else:
            if current_dtype != target_dtype:
                print(f"Error: Image {name} has bit depth {current_dtype}, expected {target_dtype}. All images must have matching bit depth.")
                return None
            if current_channels != target_channels:
                print(f"Error: Image {name} has {current_channels} color channels, expected {target_channels}. All images must have matching raw color.")
                return None

        if target_shape is None:
            target_shape = data.shape
        elif data.shape != target_shape:
            print(f"Error: Image {name} has shape {data.shape}, expected {target_shape}. All images must have the same dimensions after transformation to be stacked.")
            return None
            
        processed_images.append(data)
    
    if not processed_images:
        return None
        
    stack = np.stack(processed_images, axis=0)
    return stack

def save_stack(stack, output_path):
    """Saves the numpy stack to a .npy file."""
    np.save(output_path, stack)
    print(f"Stack saved to {output_path}")

if __name__ == "__main__":
    # Simple CLI test if run directly
    import sys
    if len(sys.argv) < 2:
        print("Usage: python image_stacker.py <directory_with_images>")
    else:
        directory = sys.argv[1]
        am = AssetManager()
        am.set_working_dir(directory)
        stack = load_and_stack_images(am)
        if stack is not None:
            print(f"Successfully created stack with shape: {stack.shape}")
            save_stack(stack, os.path.join(directory, "image_stack.npy"))
