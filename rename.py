import os

# path to dataset
base_dir = "dataset/images"

for folder in os.listdir(base_dir):
    folder_path = os.path.join(base_dir, folder)
    
    if os.path.isdir(folder_path):
        # list files inside folder
        files = os.listdir(folder_path)
        files.sort()  # sort to keep consistent order
        
        for i, filename in enumerate(files, start=1):
            ext = os.path.splitext(filename)[1]  # keep original extension (.jpg/.png)
            new_name = f"img{i}{ext}"
            old_path = os.path.join(folder_path, filename)
            new_path = os.path.join(folder_path, new_name)
            
            os.rename(old_path, new_path)
        
        print(f"Renamed files in {folder}")
