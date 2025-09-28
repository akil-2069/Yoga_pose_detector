import os
import csv

base_dir = "dataset/images"
csv_file = "labels.csv"

with open(csv_file, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["image", "label"])  # header

    for folder in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder)

        if os.path.isdir(folder_path):
            for filename in os.listdir(folder_path):
                # relative path: folder/filename
                img_path = f"{folder}/{filename}"
                writer.writerow([img_path, folder])

print(f"CSV file saved as {csv_file}")
