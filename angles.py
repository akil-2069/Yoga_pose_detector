import pandas as pd
import numpy as np

# ==================== Utils ====================

def calculate_angle(a, b, c):
    """
    Calculate the angle between three points (in degrees)
    Angle at point b
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)


def get_joint_angles(landmarks):
    """
    landmarks: 33x3 array [[x,y,z], ...]
    Returns a dict of angles for all key joints
    """
    angles = {}

    # Arms
    angles['left_elbow'] = calculate_angle(landmarks[11], landmarks[13], landmarks[15])
    angles['right_elbow'] = calculate_angle(landmarks[12], landmarks[14], landmarks[16])
    angles['left_shoulder'] = calculate_angle(landmarks[13], landmarks[11], landmarks[23])
    angles['right_shoulder'] = calculate_angle(landmarks[14], landmarks[12], landmarks[24])

    # Legs
    angles['left_knee'] = calculate_angle(landmarks[23], landmarks[25], landmarks[27])
    angles['right_knee'] = calculate_angle(landmarks[24], landmarks[26], landmarks[28])
    angles['left_hip'] = calculate_angle(landmarks[11], landmarks[23], landmarks[25])
    angles['right_hip'] = calculate_angle(landmarks[12], landmarks[24], landmarks[26])

    # Torso / Spine
    angles['spine'] = calculate_angle(landmarks[11], landmarks[23], landmarks[24])
    
    # Neck / Head
    angles['neck'] = calculate_angle(landmarks[11], landmarks[12], landmarks[0])
    angles['head'] = calculate_angle(landmarks[0], landmarks[1], landmarks[2])  # approximate
    
    return angles


# ==================== Main ====================

# File paths
input_csv = r"C:\Users\Akhilan\yoga_pose_detector\landmarks.csv"
output_csv = r"C:\Users\Akhilan\yoga_pose_detector\angles.csv"

# Read landmarks CSV
df = pd.read_csv(input_csv)

angles_data = []

for idx, row in df.iterrows():
    label = row['label']

    # Reconstruct 33x3 landmarks
    x = row[[f'x{i+1}' for i in range(33)]].values
    y = row[[f'y{i+1}' for i in range(33)]].values
    z = row[[f'z{i+1}' for i in range(33)]].values
    landmarks = np.column_stack((x, y, z))

    # Compute angles
    angles_dict = get_joint_angles(landmarks)
    angles_row = [label] + list(angles_dict.values())
    angles_data.append(angles_row)

# Header
header = ['label'] + list(angles_dict.keys())

# Save to CSV
angles_df = pd.DataFrame(angles_data, columns=header)
angles_df.to_csv(output_csv, index=False)

print(f"Angles saved to {output_csv}")
