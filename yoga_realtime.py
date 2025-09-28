"""
yoga_realtime.py

Real-time yoga pose detection + corrective instruction using MediaPipe.

Features:
- Read target angles from angles.csv (label + 11 angles)
- Compute angles from MediaPipe landmarks each frame
- Classify current pose by nearest neighbor (Euclidean on angles)
- Compare user's angles to target and show per-joint textual hints on the webcam
- Press 't' to save current angles as a new sample to angles.csv (append)
- Press 'q' to quit

CSV expected header:
label,left_elbow,right_elbow,left_shoulder,right_shoulder,left_knee,right_knee,left_hip,right_hip,spine,neck,head

If your CSV has corrupted rows, the loader will skip them with a warning.
"""

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import math
import time
from collections import defaultdict

# ---------------------------
# Utilities: angle calculation
# ---------------------------
def angle_between_points(a, b, c):
    """
    Compute the angle at point b formed by points a-b-c (in degrees).
    Points are (x,y) or (x,y,z) numpy arrays.
    """
    a = np.array(a[:2], dtype=np.float64)
    b = np.array(b[:2], dtype=np.float64)
    c = np.array(c[:2], dtype=np.float64)
    ba = a - b
    bc = c - b
    # prevent zero-length
    if np.linalg.norm(ba) < 1e-6 or np.linalg.norm(bc) < 1e-6:
        return 0.0
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle = math.degrees(math.acos(cos_angle))
    return angle

def angle_of_vector_vs_vertical(p1, p2):
    """
    Angle (degrees) between vector p1->p2 and vertical axis (y axis).
    Useful as a crude 'spine' or 'neck' inclination metric.
    """
    v = np.array(p2[:2]) - np.array(p1[:2])
    # vertical vector pointing downwards (0,1)
    vertical = np.array([0.0, 1.0])
    if np.linalg.norm(v) < 1e-6:
        return 0.0
    cos_ang = np.dot(v, vertical) / (np.linalg.norm(v) * np.linalg.norm(vertical))
    cos_ang = np.clip(cos_ang, -1.0, 1.0)
    return math.degrees(math.acos(cos_ang))

# ---------------------------
# Load angles CSV (robust)
# ---------------------------
def load_angles_csv(path="angles.csv"):
    """
    Loads CSV with header:
    label,left_elbow,right_elbow,left_shoulder,right_shoulder,left_knee,right_knee,left_hip,right_hip,spine,neck,head
    Returns: samples: list of (label, np.array(angles)), and label_means dict mapping label->mean_angles
    """
    try:
        df = pd.read_csv(path, dtype=str)
    except Exception as e:
        print(f"[load_angles_csv] Failed to read {path}: {e}")
        return [], {}

    expected_cols = ['label','left_elbow','right_elbow','left_shoulder','right_shoulder',
                     'left_knee','right_knee','left_hip','right_hip','spine','neck','head']
    # try to fix when header is fine but rows contain trailing/leading spaces
    df.columns = [c.strip() for c in df.columns]
    # drop any rows which don't have numeric values for angles
    samples = []
    for idx, row in df.iterrows():
        try:
            label = str(row['label']).strip()
            angles = []
            for col in expected_cols[1:]:
                raw = row[col]
                if pd.isna(raw):
                    raise ValueError("NaN")
                angles.append(float(str(raw).strip()))
            samples.append((label, np.array(angles, dtype=np.float32)))
        except Exception:
            print(f"[load_angles_csv] skipping malformed row {idx} in {path}")
            continue
    if not samples:
        print("[load_angles_csv] No valid samples loaded.")
        return samples, {}
    # compute means per label
    label_groups = defaultdict(list)
    for label, ang in samples:
        label_groups[label].append(ang)
    label_means = {lab: np.mean(label_groups[lab], axis=0) for lab in label_groups}
    print(f"[load_angles_csv] loaded {len(samples)} samples, {len(label_means)} labels")
    return samples, label_means

# ---------------------------
# Map joints -> mediapipe landmark triplets
# ---------------------------
# Mediapipe Pose landmark indexes reference:
# https://google.github.io/mediapipe/solutions/pose.html
# We'll use these indexes (0-based returned by mediapipe):
LM = mp.solutions.pose.PoseLandmark
JOINTS = {
    'left_elbow':  (LM.LEFT_SHOULDER, LM.LEFT_ELBOW, LM.LEFT_WRIST),
    'right_elbow': (LM.RIGHT_SHOULDER, LM.RIGHT_ELBOW, LM.RIGHT_WRIST),
    'left_shoulder': (LM.LEFT_ELBOW, LM.LEFT_SHOULDER, LM.LEFT_HIP),
    'right_shoulder': (LM.RIGHT_ELBOW, LM.RIGHT_SHOULDER, LM.RIGHT_HIP),
    'left_knee':   (LM.LEFT_HIP, LM.LEFT_KNEE, LM.LEFT_ANKLE),
    'right_knee':  (LM.RIGHT_HIP, LM.RIGHT_KNEE, LM.RIGHT_ANKLE),
    'left_hip':    (LM.LEFT_SHOULDER, LM.LEFT_HIP, LM.LEFT_KNEE),
    'right_hip':   (LM.RIGHT_SHOULDER, LM.RIGHT_HIP, LM.RIGHT_KNEE),
    # spine, neck, head - approximate estimates:
    # spine: angle between mid-shoulder -> mid-hip vs vertical
    # neck: angle between mid-shoulder -> nose vs vertical
    # head: angle between nose -> midpoint of eyes vs vertical (approx head tilt)
}

# ---------------------------
# Compute all 11 angles from landmarks
# ---------------------------
def compute_all_angles(landmarks, image_shape):
    """
    landmarks: mp.landmark list (normalized coordinates)
    returns angles np.array in order:
    [left_elbow, right_elbow, left_shoulder, right_shoulder,
     left_knee, right_knee, left_hip, right_hip, spine, neck, head]
    and angle_points: list of (x_norm, y_norm) for visualization for each of the first 8 joints
    """
    h, w = image_shape[:2]
    def lm_point(lm_enum):
        lm = landmarks[lm_enum.value]
        return np.array([lm.x, lm.y, lm.z if hasattr(lm,'z') else 0.0])

    angs = []
    angle_points = []  # normalized x,y to draw circle on
    # first 8 joints
    order = ['left_elbow','right_elbow','left_shoulder','right_shoulder',
             'left_knee','right_knee','left_hip','right_hip']
    for j in order:
        a,b,c = JOINTS[j]
        A = lm_point(a); B = lm_point(b); C = lm_point(c)
        ang = angle_between_points(A, B, C)
        angs.append(ang)
        angle_points.append((B[0], B[1]))  # center point at joint (normalized)
    # spine
    left_sh = lm_point(LM.LEFT_SHOULDER); right_sh = lm_point(LM.RIGHT_SHOULDER)
    left_hp = lm_point(LM.LEFT_HIP); right_hp = lm_point(LM.RIGHT_HIP)
    mid_sh = (left_sh + right_sh) / 2.0
    mid_hp = (left_hp + right_hp) / 2.0
    spine_ang = angle_of_vector_vs_vertical(mid_sh, mid_hp)
    angs.append(spine_ang)
    # neck: mid_sh -> nose
    nose = lm_point(LM.NOSE)
    neck_ang = angle_of_vector_vs_vertical(mid_sh, nose)
    angs.append(neck_ang)
    # head: nose -> midpoint eyes (approx)
    left_eye = lm_point(LM.LEFT_EYE)
    right_eye = lm_point(LM.RIGHT_EYE)
    mid_eyes = (left_eye + right_eye) / 2.0
    head_ang = angle_of_vector_vs_vertical(nose, mid_eyes)
    angs.append(head_ang)

    return np.array(angs, dtype=np.float32), angle_points

# ---------------------------
# Simple pose classifier (nearest neighbor on angle vector)
# ---------------------------
def classify_pose(current_angles, label_means):
    """
    current_angles: np.array(11,)
    label_means: dict label->mean_angles (np.array)
    returns (best_label, best_dist, dists_dict)
    """
    best_label = None
    best_d = float("inf")
    dists = {}
    for lab, mean_ang in label_means.items():
        d = np.linalg.norm(current_angles - mean_ang)
        dists[lab] = float(d)
        if d < best_d:
            best_d = d
            best_label = lab
    return best_label, best_d, dists

# ---------------------------
# Compare and draw instructions (adapted from your compare_pose)
# ---------------------------
def compare_pose_and_draw(image, angle_points, angle_user, angle_target):
    """
    Draws instructions onto image and returns score (lower=closer)
    angle_points: list of normalized (x,y) for joints 0..7
    angle_user: np.array(11,)
    angle_target: np.array(11,)
    Behavior:
      - For each of the 11 angles, if user deviates more than threshold show instruction
      - threshold per-joint could be tuned; here we use +/-15 degrees similar to your code
    """
    height, width, _ = image.shape
    angle_user = np.array(angle_user)
    angle_target = np.array(angle_target)
    threshold = 15.0
    issues = []
    # text area (white background)
    cv2.rectangle(image,(0,0),(420,420),(255,255,255),-1)
    cv2.putText(image, "Pose correction (press 't' to save, 'q' to quit)", (10,20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
    y = 40
    # Map to human readable messages similar to yours
    messages = [
        ("left elbow", "Extend the right arm at elbow", "Fold the right arm at elbow"),  # user had indices swapped? keep consistent
        ("right elbow", "Extend the left arm at elbow", "Fold the left arm at elbow"),
        ("left shoulder", "Lift your right arm", "Put your arm down a little"),
        ("right shoulder", "Lift your left arm", "Put your arm down a little"),
        ("left knee", "Extend the angle at right hip", "Reduce the angle at right hip"),
        ("right knee", "Extend the angle at left hip", "Reduce the angle at left hip"),
        ("left hip", "Extend the angle of right knee", "Reduce the angle of right knee"),
        ("right hip", "Extend the angle at left knee", "Reduce the angle at left knee"),
        ("spine", "Straighten your spine", "Straighten your spine"),
        ("neck", "Lift your neck", "Lower your neck"),
        ("head", "Keep head upright", "Keep head upright"),
    ]
    # We'll compare all 11
    score = 0
    for i in range(len(angle_user)):
        diff = angle_user[i] - angle_target[i]
        if diff < -threshold:
            # user angle smaller than target -> extend
            msg = messages[i][1]
            issues.append((i, msg))
            score += 1
            cv2.putText(image, msg, (10,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)
        elif diff > threshold:
            msg = messages[i][2]
            issues.append((i, msg))
            score += 1
            cv2.putText(image, msg, (10,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)
        else:
            # good for this joint
            pass
        y += 20
    # draw circles on problematic joints (only for first 8 joints we have angle_points)
    for (idx, msg) in issues:
        if idx < len(angle_points):
            x = int(angle_points[idx][0] * width)
            y2 = int(angle_points[idx][1] * height)
            cv2.circle(image, (x,y2), 18, (0,0,255), 3)
    # Score display (lower is better)
    cv2.putText(image, f"Score (num mismatches): {score}", (10,400),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,153,0), 2, cv2.LINE_AA)
    return score, issues

# ---------------------------
# Append a sample to CSV
# ---------------------------
def append_sample_to_csv(path, label, angles):
    """
    Append a new sample (label + angles) to CSV. Creates file with header if not exists.
    """
    cols = ['label','left_elbow','right_elbow','left_shoulder','right_shoulder',
            'left_knee','right_knee','left_hip','right_hip','spine','neck','head']
    row = [label] + [float(x) for x in angles.tolist()]
    try:
        df = pd.DataFrame([row], columns=cols)
        with open(path, 'a', newline='') as f:
            df.to_csv(f, header=f.tell()==0, index=False)
        print(f"[append] appended new sample for label '{label}'")
    except Exception as e:
        print(f"[append] failed to append: {e}")

# ---------------------------
# Main real-time loop
# ---------------------------
def run_realtime(csv_path="angles.csv"):
    samples, label_means = load_angles_csv(csv_path)

    # default label when saving new sample
    default_label = "downdog"

    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        last_class = None
        last_dist = None
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Flip for mirror effect
            frame = cv2.flip(frame, 1)
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                # compute angles
                try:
                    angles, angle_points = compute_all_angles(landmarks, frame.shape)
                except Exception as e:
                    print(f"[compute] error computing angles: {e}")
                    angles = np.zeros(11, dtype=np.float32)
                    angle_points = [(0.5,0.5)]*8
                # classify
                if label_means:
                    best_label, best_d, dists = classify_pose(angles, label_means)
                    last_class = best_label
                    last_dist = best_d
                    # draw landmark skeleton
                    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                    # draw instruction overlay
                    score, issues = compare_pose_and_draw(frame, angle_points, angles, label_means[best_label])
                    # text: detected label
                    cv2.putText(frame, f"Detected: {best_label} (dist {best_d:.1f})", (10,440),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2, cv2.LINE_AA)
                else:
                    cv2.putText(frame, "No training data loaded. Press 't' to save current as sample.", (10,20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2, cv2.LINE_AA)
                    # still allow draw
                    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            else:
                cv2.putText(frame, "No pose detected", (10,20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2, cv2.LINE_AA)

            cv2.imshow("Yoga Detection", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('t'):
                # save current sample to CSV
                if results.pose_landmarks:
                    angles_to_save, _ = compute_all_angles(results.pose_landmarks.landmark, frame.shape)
                    append_sample_to_csv(csv_path, default_label, angles_to_save)
                    # reload samples & label_means
                    samples, label_means = load_angles_csv(csv_path)
                else:
                    print("[info] No pose to save at this moment.")
            # quick way to change default label via keyboard numbers? keep simple
            # (advanced UX: popup to type label - keep code minimal)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_realtime("angles.csv")
