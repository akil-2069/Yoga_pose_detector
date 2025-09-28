import cv2
import mediapipe as mp
import os
import csv
import numpy as np
from tqdm import tqdm

# MediaPipe pose setup
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

# Paths
dataset_path = r"C:\Users\Akhilan\yoga_pose_detector\dataset\images"
output_csv = r"C:\Users\Akhilan\yoga_pose_detector\landmarks.csv"

# Function to extract landmarks from image
def extract_landmarks(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    if results.pose_landmarks:
        landmarks = []
        for lm in results.pose_landmarks.landmark:
            landmarks.append([lm.x, lm.y, lm.z])
        return np.array(landmarks).flatten()  # Flatten to 1D array
    else:
        return None

# Collect data
data = []

for asana in os.listdir(dataset_path):
    asana_folder = os.path.join(dataset_path, asana)
    if not os.path.isdir(asana_folder):
        continue
    for img_file in tqdm(os.listdir(asana_folder), desc=f"Processing {asana}"):
        if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(asana_folder, img_file)
            image = cv2.imread(img_path)
            landmarks = extract_landmarks(image)
            if landmarks is not None:
                data.append([asana] + landmarks.tolist())

# Write to CSV
with open(output_csv, 'w', newline='') as f:
    writer = csv.writer(f)
    # Header: label + flattened landmark coordinates
    header = ['label'] + [f'x{i+1}' for i in range(33)] + [f'y{i+1}' for i in range(33)] + [f'z{i+1}' for i in range(33)]
    writer.writerow(header)
    for row in data:
        writer.writerow(row)

print(f"Landmarks saved to {output_csv}")
