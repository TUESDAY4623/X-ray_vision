# import os
# from PIL import Image
# import glob

# # Replace with your actual image directories (train/val/test folders)
# image_dirs = [
#     r"E:\ai_model\project\X-ray_data\train\*",  # Use glob patterns for all images
#     r"E:\ai_model\project\X-ray_data\test\*",  # Use glob patterns for all images
#     r"E:\ai_model\project\X-ray_data\val\*",
#     # Add more paths as needed
# ]

# image_paths = []
# for pattern in image_dirs:
#     image_paths.extend(glob.glob(pattern, recursive=True))  # Finds all .jpg, .png etc.

# print(f"Scanning {len(image_paths)} images...")

# deleted = 0
# for img_path in image_paths:
#     try:
#         img = Image.open(img_path)
#         img.verify()  # Integrity check
#         img.close()
#     except Exception as e:
#         print(f"Deleting corrupted: {img_path}")
#         os.remove(img_path)
#         deleted += 1

# print(f"Deleted {deleted} corrupted images. Ready to train!")

# import os
# from PIL import Image
# from pathlib import Path

# # Define your dataset root directory
# dataset_root = r"E:\ai_model\project\X-ray_data"  # Change this to YOUR path!

# print(f"Scanning for images in: {dataset_root}")

# deleted = 0
# scanned = 0

# # Recursively find ALL image files
# for root, dirs, files in os.walk(dataset_root):
#     for file in files:
#         if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
#             img_path = os.path.join(root, file)
#             scanned += 1
            
#             try:
#                 img = Image.open(img_path)
#                 img.verify()
#                 img.close()
#             except Exception as e:
#                 print(f"🗑️  Deleting: {img_path}")
#                 try:
#                     os.remove(img_path)
#                     deleted += 1
#                 except:
#                     pass

# print(f"\n✅ Scanned {scanned} images")
# print(f"🗑️  Deleted {deleted} corrupted images")
import os
import config
from PIL import Image
from datetime import datetime
import json

dataset_root = config.DATA_DIR  # auto-resolves relative to config.py location
log_file = "dataset_cleaning_report.json"

print(f"🔍 Scanning for images in: {dataset_root}")
print("=" * 70)

deleted = 0
scanned = 0
passed = 0
failed = 0

# Track all results
report = {
    "timestamp": datetime.now().isoformat(),
    "dataset_root": dataset_root,
    "healthy_images": [],
    "corrupted_images": [],
    "summary": {}
}

# Walk through all directories and files
for root, dirs, files in os.walk(dataset_root):
    class_name = os.path.basename(root)
    
    for file in files:
        # Only process image files
        if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
            img_path = os.path.join(root, file)
            scanned += 1
            
            # Progress indicator
            if scanned % 1000 == 0:
                print(f"⏳ Progress: Scanned {scanned} images...")
            
            try:
                img = Image.open(img_path)
                img.verify()
                img.close()
                passed += 1
                
                # Log healthy images
                report["healthy_images"].append({
                    "path": img_path,
                    "class": class_name,
                    "status": "✅ PASSED"
                })
                
                # Print every 5000 passed images
                if passed % 5000 == 0:
                    print(f"✅ {passed} healthy images verified so far...")
                
            except Exception as e:
                failed += 1
                deleted += 1
                
                print(f"❌ CORRUPTED: {img_path}")
                print(f"   └─ Error: {str(e)[:80]}")
                print(f"   └─ Class: {class_name}")
                
                # Log corrupted images
                report["corrupted_images"].append({
                    "path": img_path,
                    "class": class_name,
                    "error": str(e),
                    "status": "❌ DELETED"
                })
                
                try:
                    os.remove(img_path)
                    print(f"   └─ ✅ Successfully removed\n")
                except PermissionError as pe:
                    print(f"   └─ ⚠️  Permission denied - could not delete\n")
                    report["corrupted_images"][-1]["status"] = "⚠️  PERMISSION DENIED"

# Summary Statistics
print("\n" + "=" * 70)
print("📊 DATASET CLEANING REPORT")
print("=" * 70)
print(f"✅ Total Scanned:      {scanned} images")
print(f"✅ Healthy Images:     {passed} images")
print(f"❌ Corrupted Images:   {failed} images")
print(f"🗑️  Deleted:           {deleted} images")
print(f"📈 Health Score:       {(passed/scanned)*100:.2f}%")

# Class-wise breakdown
print("\n📁 Per-Class Analysis:")
class_stats = {}
for record in report["healthy_images"] + report["corrupted_images"]:
    class_name = record["class"]
    status = "healthy" if "✅" in record["status"] else "corrupted"
    
    if class_name not in class_stats:
        class_stats[class_name] = {"healthy": 0, "corrupted": 0}
    
    class_stats[class_name][status] += 1

for class_name, stats in sorted(class_stats.items()):
    total = stats["healthy"] + stats["corrupted"]
    health = (stats["healthy"] / total * 100) if total > 0 else 0
    print(f"   {class_name:25} ✅ {stats['healthy']:6} | ❌ {stats['corrupted']:6} | Health: {health:5.1f}%")

# Save detailed report
report["summary"] = {
    "total_scanned": scanned,
    "healthy": passed,
    "corrupted": failed,
    "deleted": deleted,
    "health_score_percent": (passed/scanned)*100 if scanned > 0 else 0,
    "class_statistics": class_stats
}

with open(log_file, 'w') as f:
    json.dump(report, f, indent=2)

print(f"\n📄 Detailed report saved to: {log_file}")
print("=" * 70)
print("✅ Ready to train!\n")

