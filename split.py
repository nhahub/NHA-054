import os
import shutil
import random
import yaml
from sklearn.model_selection import train_test_split

# --- Configuration ---
# *** EDITED: Explicitly defined all three ratios ***
TRAIN_RATIO = 0.75
VAL_RATIO = 0.15
TEST_RATIO = 0.10
RANDOM_SEED = 42 # For reproducible splits

# Check if ratios sum to 1.0
if round(TRAIN_RATIO + VAL_RATIO + TEST_RATIO, 5) != 1.0:
    print(f"Error: Ratios must sum to 1.0. Current sum is {TRAIN_RATIO + VAL_RATIO + TEST_RATIO}")
    exit()
# ---------------------

def get_subdirectories(folder):
    """Helper function to get a list of immediate subdirectories."""
    try:
        return [f.path for f in os.scandir(folder) if f.is_dir()]
    except FileNotFoundError:
        print(f"Error: Base directory not found: {folder}")
        return []

def create_output_structure(base_dir):
    """Creates the train/val/test images and labels folders."""
    paths = {}
    for split in ['train', 'val', 'test']:
        paths[f'{split}_images'] = os.path.join(base_dir, split, 'images')
        paths[f'{split}_labels'] = os.path.join(base_dir, split, 'labels')
        
        os.makedirs(paths[f'{split}_images'], exist_ok=True)
        os.makedirs(paths[f'{split}_labels'], exist_ok=True)
    return paths

def build_global_classes(material_folders):
    """Reads all classes.txt files and builds a single global list."""
    global_classes = []
    for material_folder in material_folders:
        classes_file_path = os.path.join(material_folder, "classes.txt")
        if os.path.exists(classes_file_path):
            try:
                with open(classes_file_path, 'r', encoding='utf-8') as f:
                    local_classes = [line.strip() for line in f if line.strip()]
                    for cls_name in local_classes:
                        if cls_name not in global_classes:
                            global_classes.append(cls_name)
            except Exception as e:
                print(f"Error reading {classes_file_path}: {e}")
    return global_classes

def get_local_to_global_map(material_folder, global_classes_list):
    """Builds a {local_index: global_index} map for a single material."""
    index_mapping = {}
    local_classes_path = os.path.join(material_folder, "classes.txt")
    if not os.path.exists(local_classes_path):
        return None
        
    try:
        with open(local_classes_path, 'r', encoding='utf-8') as f:
            local_classes = [line.strip() for line in f if line.strip()]
        
        for local_index, class_name in enumerate(local_classes):
            try:
                global_index = global_classes_list.index(class_name)
                index_mapping[local_index] = global_index
            except ValueError:
                print(f"  - Warning: Class '{class_name}' from {material_folder} not in global list.")
        return index_mapping
    except Exception as e:
        print(f"  - Error reading {local_classes_path}: {e}")
        return None

def process_files_split(file_basenames, material_folder, index_mapping, dest_images_dir, dest_labels_dir):
    """Copies images and re-mapped labels for a given split (train, val, or test)."""
    source_images_dir = os.path.join(material_folder, "images")
    source_labels_dir = os.path.join(material_folder, "labels")
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    files_processed = 0
    
    for base_name in file_basenames:
        # 1. Copy Image
        image_copied = False
        for ext in image_extensions:
            image_filename = base_name + ext
            source_image_path = os.path.join(source_images_dir, image_filename)
            if os.path.exists(source_image_path):
                dest_image_path = os.path.join(dest_images_dir, image_filename)
                try:
                    shutil.copy2(source_image_path, dest_image_path)
                    image_copied = True
                    break
                except Exception as e:
                    print(f"  - Error copying image {source_image_path}: {e}")
        
        if not image_copied:
            print(f"  - Warning: Could not find image file for {base_name}")
            continue

        # 2. Find, Remap, and Copy Label
        label_filename = base_name + ".txt"
        source_label_path = os.path.join(source_labels_dir, label_filename)
        dest_label_path = os.path.join(dest_labels_dir, label_filename)

        if os.path.exists(source_label_path):
            new_label_content = []
            try:
                with open(source_label_path, 'r', encoding='utf-8') as f_in:
                    for line in f_in:
                        parts = line.strip().split()
                        if not parts: continue
                        
                        local_index = int(parts[0])
                        if local_index in index_mapping:
                            global_index = index_mapping[local_index]
                            new_line = f"{global_index} {' '.join(parts[1:])}"
                            new_label_content.append(new_line)
                        else:
                            print(f"  - Warning: Invalid class index {local_index} in {source_label_path}")
                
                with open(dest_label_path, 'w', encoding='utf-8') as f_out:
                    f_out.write("\n".join(new_label_content))
                files_processed += 1
                
            except Exception as e:
                print(f"  - Error processing label {source_label_path}: {e}")
        else:
            # Create an empty label file for unlabeled (negative) images
            try:
                with open(dest_label_path, 'w') as f_out:
                    pass # Create empty file
                files_processed += 1
            except Exception as e:
                print(f"  - Error creating empty label for {dest_label_path}: {e}")

    return files_processed

def main():
    # --- CONFIGURE YOUR PATHS HERE ---
    
    # Path to the main folder containing your 7 material folders
    base_data_dir = r"G:\split\all_data"
    
    # Path where the new 'train', 'val', 'test' folders will be created
    output_split_dir = r"G:\split\all_data\splitted"
    
    # --- END OF CONFIGURATION ---

    print(f"Starting balanced {TRAIN_RATIO*100}% / {VAL_RATIO*100}% / {TEST_RATIO*100}% split...")
    print(f"Source: {base_data_dir}")
    print(f"Destination: {output_split_dir}\n")
    
    material_folders = get_subdirectories(base_data_dir)
    if not material_folders: return

    # 1. Build Global Class List
    print("Building global class list...")
    global_classes = build_global_classes(material_folders)
    if not global_classes:
        print("Error: No classes.txt files found. Exiting.")
        return
    print(f"Found {len(global_classes)} unique classes total.\n")

    # 2. Create output folder structure
    dest_paths = create_output_structure(output_split_dir)
    
    total_train_files = 0
    total_val_files = 0
    total_test_files = 0
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

    # 3. Iterate through each material, split, copy, and remap
    for material_folder in material_folders:
        material_name = os.path.basename(material_folder)
        print(f"--- Processing: {material_name} ---")

        # A. Get the index mapping
        index_mapping = get_local_to_global_map(material_folder, global_classes)
        if index_mapping is None:
            print("  - Skipping (no classes.txt or error).")
            continue
            
        # B. Get list of all images
        images_dir = os.path.join(material_folder, "images")
        if not os.path.exists(images_dir):
            print("  - Skipping (no 'images' folder).")
            continue
            
        all_image_basenames = []
        for f in os.listdir(images_dir):
            file_name, file_ext = os.path.splitext(f)
            if file_ext.lower() in image_extensions:
                all_image_basenames.append(file_name)
        
        if not all_image_basenames:
            print("  - Skipping (no images found).")
            continue
            
        print(f"  - Found {len(all_image_basenames)} images.")

        # --- *** EDITED: Updated split logic to use all 3 ratios *** ---
        # C. Perform the 75/15/10 split
        
        # First split: 75% train, 25% (val + test)
        train_files, remaining_files = train_test_split(
            all_image_basenames,
            train_size=TRAIN_RATIO,
            random_state=RANDOM_SEED
        )
        
        # Calculate ratio for second split
        # We need to split the 'remaining' (25%) into val (15%) and test (10%)
        # The ratio of val within the remaining set is: VAL_RATIO / (VAL_RATIO + TEST_RATIO)
        # e.g., 0.15 / (0.15 + 0.10) = 0.15 / 0.25 = 0.6
        val_of_remaining_ratio = VAL_RATIO / (VAL_RATIO + TEST_RATIO)
        
        # Second split
        val_files, test_files = train_test_split(
            remaining_files,
            train_size=val_of_remaining_ratio, 
            random_state=RANDOM_SEED # Use same seed
        )
        # --- *** END OF EDIT *** ---
        
        print(f"  - Splitting: {len(train_files)} train / {len(val_files)} val / {len(test_files)} test")

        # D. Process and copy files to their final destinations
        # Process TRAIN
        train_count = process_files_split(
            train_files, material_folder, index_mapping, 
            dest_paths['train_images'], dest_paths['train_labels']
        )
        # Process VAL
        val_count = process_files_split(
            val_files, material_folder, index_mapping, 
            dest_paths['val_images'], dest_paths['val_labels']
        )
        # Process TEST
        test_count = process_files_split(
            test_files, material_folder, index_mapping, 
            dest_paths['test_images'], dest_paths['test_labels']
        )
        
        total_train_files += train_count
        total_val_files += val_count
        total_test_files += test_count

    # 4. Create the final data.yaml file
    yaml_path = os.path.join(output_split_dir, "data.yaml")
    
    train_path = os.path.join('train', 'images')
    val_path = os.path.join('val', 'images')
    test_path = os.path.join('test', 'images')
    
    yaml_content = {
        'train': train_path,
        'val': val_path,
        'test': test_path,
        'nc': len(global_classes),
        'names': global_classes
    }
    
    try:
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(yaml_content, f, sort_keys=False, default_flow_style=False)
        print(f"\nSuccessfully created data.yaml at {yaml_path}")
    except Exception as e:
        print(f"\nError creating data.yaml: {e}")

    # 5. Final Report
    print("\n--- Split Complete! ---")
    print(f"Total training images/labels: {total_train_files}")
    print(f"Total validation images/labels: {total_val_files}")
    print(f"Total test images/labels: {total_test_files}")
    print(f"Total files: {total_train_files + total_val_files + total_test_files}")
    print(f"Dataset saved to: {output_split_dir}")
    print("\nYour dataset is ready for training. Point your model to the 'data.yaml' file.")


if __name__ == "__main__":
    main()