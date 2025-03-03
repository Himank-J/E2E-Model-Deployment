import os
import shutil
from pathlib import Path
import json

def combine_datasets(intel_path, sports_path, main_data_path):
    """Combine Intel and Sports datasets into a single dataset."""
    
    # Create main_data directory structure
    splits = ['train', 'test', 'valid']
    for split in splits:
        Path(os.path.join(main_data_path, split)).mkdir(parents=True, exist_ok=True)
    
    # Process both datasets
    datasets = [
        ('intel', intel_path),
        ('sports', sports_path)
    ]
    
    # Calculate class weights
    class_counts = {}
    for dataset_name, dataset_path in datasets:
        split_path = os.path.join(dataset_path, 'train')
        for class_dir in os.listdir(split_path):
            if os.path.isdir(os.path.join(split_path, class_dir)):
                img_count = len(os.listdir(os.path.join(split_path, class_dir)))
                class_counts[class_dir] = img_count
    
    # Save class counts to file
    with open(os.path.join(main_data_path, 'class_counts.json'), 'w') as f:
        json.dump(class_counts, f)
    
    print("\nClass distribution:")
    for class_name, count in class_counts.items():
        print(f"{class_name}: {count} images")
    
    for dataset_name, dataset_path in datasets:
        print(f"Processing {dataset_name} dataset...")
        
        # Copy each split
        for split in splits:
            src_split_path = os.path.join(dataset_path, split)
            
            # Get all class directories
            class_dirs = [d for d in os.listdir(src_split_path) 
                        if os.path.isdir(os.path.join(src_split_path, d))]
            
            # Copy each class directory with prefix to avoid name conflicts
            for class_dir in class_dirs:
                src_class_path = os.path.join(src_split_path, class_dir)
                dst_class_name = f"{class_dir}" 
                dst_class_path = os.path.join(main_data_path, split, dst_class_name)
                
                if os.path.exists(dst_class_path):
                    shutil.rmtree(dst_class_path)
                shutil.copytree(src_class_path, dst_class_path)
                
            print(f"Completed {split} split for {dataset_name}")

def main():
    # Define paths
    base_path = os.path.join(os.getcwd(), 'data')
    intel_path = os.path.join(base_path, 'intel')
    sports_path = os.path.join(base_path, 'sports')
    main_data_path = os.path.join(base_path, 'main_data')
    
    # Combine datasets
    combine_datasets(intel_path, sports_path, main_data_path)
    print(f"Combined dataset created at: {main_data_path}")

if __name__ == "__main__":
    main() 