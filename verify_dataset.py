"""
Dataset Path Verification Script
Checks if all dataset paths in data.yaml are correct and accessible.
"""

import os
from pathlib import Path
import yaml

# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent
DATA_YAML = SCRIPT_DIR / "data.yaml"

print("=" * 70)
print(" Dataset Path Verification")
print("=" * 70)

if not DATA_YAML.exists():
    print(f"❌ data.yaml not found at: {DATA_YAML}")
    exit(1)

print(f"\n✅ Found data.yaml at: {DATA_YAML}")

# Read data.yaml
with open(DATA_YAML, 'r') as f:
    data = yaml.safe_load(f)

print("\n📋 Dataset Configuration:")
print(f"   Number of classes: {data.get('nc', 'N/A')}")
print(f"   Classes: {', '.join(data.get('names', []))}")

# Check paths
paths_to_check = {
    'train': data.get('train'),
    'val': data.get('val'),
    'test': data.get('test')
}

print("\n🔍 Checking Dataset Paths:")
all_valid = True

for split, path in paths_to_check.items():
    if not path:
        print(f"   ❌ {split}: Path not specified in data.yaml")
        all_valid = False
        continue
    
    # Convert to Path object for easier handling
    path_obj = Path(path)
    
    # Check if path exists
    if path_obj.exists():
        # Count images
        image_files = list(path_obj.glob("*.jpg")) + list(path_obj.glob("*.png")) + list(path_obj.glob("*.jpeg"))
        print(f"   ✅ {split}: {path}")
        print(f"      Images found: {len(image_files)}")
        
        # Check for corresponding labels
        labels_path = path_obj.parent / "labels"
        if labels_path.exists():
            label_files = list(labels_path.glob("*.txt"))
            print(f"      Labels found: {len(label_files)}")
            
            if len(image_files) != len(label_files):
                print(f"      ⚠️ Warning: Image count ({len(image_files)}) != Label count ({len(label_files)})")
        else:
            print(f"      ⚠️ Warning: Labels directory not found at {labels_path}")
    else:
        print(f"   ❌ {split}: Path does not exist: {path}")
        all_valid = False
        
        # Suggest correct path
        # Try to find the directory
        if "images" in str(path):
            parent_dir = Path(path).parent.parent.parent
            if parent_dir.exists():
                print(f"      💡 Suggestion: Check if dataset is in: {parent_dir}")

print("\n" + "=" * 70)
if all_valid:
    print("✅ All dataset paths are valid!")
    print("   You can proceed with training.")
else:
    print("❌ Some dataset paths are invalid!")
    print("   Please fix the paths in data.yaml before training.")
print("=" * 70)

