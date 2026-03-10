"""
Data Organizer for Multi-Disease X-Ray Detection System
Extracts ZIP files and organizes datasets into train/val/test structure
"""
import os
import zipfile
import shutil
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
import config

def extract_zip_files(base_dir='X-ray_data'):
    """Extract all ZIP files in the dataset directory"""
    print("🔍 Searching for ZIP files...")
    zip_files = []
    
    # Search in multiple locations
    search_dirs = [
        'dataset',           # User's dataset folder
        'X-ray_data',        # Original location
        '.',                  # Current directory (project root)
        'data',              # Alternative data folder
    ]
    
    # Find all ZIP files in all search directories
    for search_dir in search_dirs:
        if os.path.exists(search_dir):
            print(f"   Checking: {search_dir}/")
            for root, dirs, files in os.walk(search_dir):
                for file in files:
                    if file.lower().endswith('.zip'):
                        zip_path = os.path.join(root, file)
                        # Avoid duplicates
                        if zip_path not in zip_files:
                            zip_files.append(zip_path)
                            print(f"      ✅ Found: {file} in {root}")
        else:
            print(f"   ⚠️  {search_dir}/ not found, skipping...")
    
    print(f"\n📦 Found {len(zip_files)} ZIP files total")
    
    # Create base directory if it doesn't exist
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
        print(f"\n📁 Created directory: {base_dir}")
    
    # Extract each ZIP file
    extracted_dirs = []
    for zip_path in zip_files:
        zip_name = os.path.basename(zip_path)
        print(f"\n📂 Extracting: {zip_name}")
        
        # Extract to base_dir with cleaned name
        extract_dir = os.path.join(base_dir, zip_name.replace('.zip', '').replace('&', '_and_'))
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # Get list of files to check size
                file_list = zip_ref.namelist()
                print(f"   📦 Contains {len(file_list)} files")
                
                zip_ref.extractall(extract_dir)
            print(f"   ✅ Extracted to: {extract_dir}")
            extracted_dirs.append(extract_dir)
        except Exception as e:
            print(f"   ❌ Error extracting {zip_path}: {str(e)}")
            import traceback
            print(f"   Details: {traceback.format_exc()[:200]}")
    
    return extracted_dirs

def read_metadata(base_dir='X-ray_data'):
    """Read metadata from CSV and TXT files"""
    print("\n📋 Reading metadata files...")
    metadata = {}
    
    metadata_dir = os.path.join(base_dir, 'Metadata')
    if os.path.exists(metadata_dir):
        for file in os.listdir(metadata_dir):
            file_path = os.path.join(metadata_dir, file)
            
            if file.endswith('.csv'):
                try:
                    df = pd.read_csv(file_path)
                    print(f"   ✅ Loaded: {file} ({len(df)} rows)")
                    metadata[file] = df
                except Exception as e:
                    print(f"   ❌ Error reading {file}: {str(e)}")
            
            elif file.endswith('.txt'):
                try:
                    with open(file_path, 'r') as f:
                        lines = f.readlines()
                    print(f"   ✅ Loaded: {file} ({len(lines)} lines)")
                    metadata[file] = lines
                except Exception as e:
                    print(f"   ❌ Error reading {file}: {str(e)}")
    
    return metadata

def map_image_to_label(image_path, metadata):
    """Map image filename to disease label based on metadata and path"""
    filename = os.path.basename(image_path)
    filename_lower = filename.lower()
    path_lower = image_path.lower()
    
    # Check parent directory names (most reliable)
    path_parts = path_lower.split(os.sep)
    for part in path_parts:
        # Handle specific ZIP file names from user's dataset
        # Note: Handle misspellings like "facture" instead of "fracture"
        if 'bone_facture' in part or 'bone_fracture' in part or 'fracture' in part or 'fract' in part:
            return 'BONE_FRACTURE'
        if 'bone_supression' in part or 'bone_suppression' in part or 'suppression' in part or 'suppress' in part:
            return 'BONE_SUPPRESSION'
        if 'brain_tumor' in part or ('brain' in part and 'tumor' in part):
            return 'BRAIN_TUMOR'
        if 'chest_nih' in part or ('chest' in part and 'nih' in part):
            return 'CHEST_NIH_ABNORMAL'
        # Handle combined ZIP: pnemonia&COVID19 (check for both)
        if 'pnemonia' in part and 'covid' in part:
            # If both are in path, check filename or subdirectory for more specific match
            if 'covid' in filename_lower:
                return 'COVID19'
            else:
                return 'PNEUMONIA'  # Default to pneumonia if ambiguous
        if 'pnemonia' in part or 'pneumonia' in part or 'pneum' in part:
            return 'PNEUMONIA'
        if 'covid' in part or 'corona' in part:
            return 'COVID19'
        if 'normal' in part or 'healthy' in part:
            return 'NORMAL'
    
    # Check subdirectories in path (for nested structures)
    if 'normal' in path_lower or 'healthy' in path_lower:
        return 'NORMAL'
    if 'brain' in path_lower or 'tumor' in path_lower:
        return 'BRAIN_TUMOR'
    if 'pneumonia' in path_lower or 'pnemonia' in path_lower:
        return 'PNEUMONIA'
    if 'covid' in path_lower:
        return 'COVID19'
    if 'fracture' in path_lower or 'fract' in path_lower:
        return 'BONE_FRACTURE'
    if 'suppression' in path_lower or 'suppress' in path_lower:
        return 'BONE_SUPPRESSION'
    if 'chest' in path_lower and 'nih' in path_lower:
        return 'CHEST_NIH_ABNORMAL'
    
    # Check filename
    if 'brain' in filename_lower or 'tumor' in filename_lower:
        return 'BRAIN_TUMOR'
    if 'pneumonia' in filename_lower or 'pnemonia' in filename_lower or 'pneum' in filename_lower:
        return 'PNEUMONIA'
    if 'covid' in filename_lower or 'corona' in filename_lower:
        return 'COVID19'
    if 'fracture' in filename_lower or 'fract' in filename_lower:
        return 'BONE_FRACTURE'
    if 'suppression' in filename_lower or 'suppress' in filename_lower:
        return 'BONE_SUPPRESSION'
    if 'chest' in filename_lower and ('abnormal' in filename_lower or 'nih' in filename_lower):
        return 'CHEST_NIH_ABNORMAL'
    if 'normal' in filename_lower or 'healthy' in filename_lower:
        return 'NORMAL'
    
    # Default to NORMAL if uncertain
    return 'NORMAL'

def is_valid_image(file_path):
    """Check if file is a valid image"""
    try:
        img = Image.open(file_path)
        img.verify()
        return True
    except Exception:
        return False

def collect_all_images(base_dir='X-ray_data'):
    """Collect all images from extracted datasets"""
    print("\n🖼️  Collecting all images...")
    all_images = []
    
    # Search in multiple locations
    search_dirs = [
        base_dir,
        'dataset',
        '.',
        'data',
    ]
    
    searched_paths = set()  # Track searched paths to avoid duplicates
    
    for search_dir in search_dirs:
        if os.path.exists(search_dir):
            print(f"   Searching in: {search_dir}/")
            for root, dirs, files in os.walk(search_dir):
                # Skip organized directories
                if 'train' in root or 'val' in root or 'test' in root:
                    continue
                
                # Skip if already searched
                if root in searched_paths:
                    continue
                
                searched_paths.add(root)
                
                for file in files:
                    if any(file.lower().endswith(ext) for ext in config.SUPPORTED_FORMATS):
                        file_path = os.path.join(root, file)
                        if is_valid_image(file_path):
                            label = map_image_to_label(file_path, {})
                            all_images.append((file_path, label))
    
    print(f"   ✅ Found {len(all_images)} valid images")
    return all_images

def organize_dataset(images, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    """Organize images into train/val/test directories"""
    print("\n📁 Organizing dataset into train/val/test...")
    
    # Create directory structure
    for split in ['train', 'val', 'test']:
        for class_name in config.CLASSES:
            dir_path = os.path.join(config.BASE_DIR, split, class_name)
            os.makedirs(dir_path, exist_ok=True)
    
    # Group images by class
    class_images = {class_name: [] for class_name in config.CLASSES}
    for img_path, label in images:
        if label in class_images:
            class_images[label].append(img_path)
    
    # Split each class into train/val/test
    stats = {}
    for class_name, img_list in class_images.items():
        np.random.shuffle(img_list)
        n_total = len(img_list)
        
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        n_test = n_total - n_train - n_val
        
        # Copy images to respective directories
        for i, img_path in enumerate(img_list):
            if i < n_train:
                dest_dir = os.path.join(config.TRAIN_DIR, class_name)
            elif i < n_train + n_val:
                dest_dir = os.path.join(config.VAL_DIR, class_name)
            else:
                dest_dir = os.path.join(config.TEST_DIR, class_name)
            
            filename = os.path.basename(img_path)
            dest_path = os.path.join(dest_dir, filename)
            
            # Handle duplicate filenames
            counter = 1
            while os.path.exists(dest_path):
                name, ext = os.path.splitext(filename)
                dest_path = os.path.join(dest_dir, f"{name}_{counter}{ext}")
                counter += 1
            
            shutil.copy2(img_path, dest_path)
        
        stats[class_name] = {
            'train': n_train,
            'val': n_val,
            'test': n_test,
            'total': n_total
        }
    
    return stats

def print_statistics(stats):
    """Print dataset statistics"""
    print("\n" + "="*60)
    print("📊 DATASET STATISTICS")
    print("="*60)
    print(f"{'Class':<25} {'Train':<10} {'Val':<10} {'Test':<10} {'Total':<10}")
    print("-"*60)
    
    total_train = total_val = total_test = total_all = 0
    
    for class_name in config.CLASSES:
        if class_name in stats:
            s = stats[class_name]
            print(f"{class_name:<25} {s['train']:<10} {s['val']:<10} {s['test']:<10} {s['total']:<10}")
            total_train += s['train']
            total_val += s['val']
            total_test += s['test']
            total_all += s['total']
    
    print("-"*60)
    print(f"{'TOTAL':<25} {total_train:<10} {total_val:<10} {total_test:<10} {total_all:<10}")
    print("="*60)
    
    # Calculate percentages
    print(f"\n📈 Split Ratios:")
    print(f"   Train: {total_train/total_all*100:.1f}% ({total_train} images)")
    print(f"   Val:   {total_val/total_all*100:.1f}% ({total_val} images)")
    print(f"   Test:  {total_test/total_all*100:.1f}% ({total_test} images)")

def main():
    """Main function to organize dataset"""
    print("="*60)
    print("🏥 MULTI-DISEASE X-RAY DATA ORGANIZER")
    print("="*60)
    
    # Step 1: Extract ZIP files
    extracted_dirs = extract_zip_files()
    
    # Step 2: Read metadata
    metadata = read_metadata()
    
    # Step 3: Collect all images
    images = collect_all_images()
    
    if len(images) == 0:
        print("\n❌ No images found! Please check your dataset structure.")
        return
    
    # Step 4: Organize dataset
    stats = organize_dataset(images)
    
    # Step 5: Print statistics
    print_statistics(stats)
    
    print("\n✅ Dataset organization complete!")
    print(f"📁 Organized data saved to: {config.BASE_DIR}/")

if __name__ == "__main__":
    main()
