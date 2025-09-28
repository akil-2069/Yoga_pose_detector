import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import joblib

# ==================== Load Data ====================

angles_csv = r"C:\Users\Akhilan\yoga_pose_detector\angles.csv"
df = pd.read_csv(angles_csv)

X = df.drop('label', axis=1).values
y = df['label'].values

# ==================== Encode Labels ====================

le = LabelEncoder()
y_encoded = le.fit_transform(y)
num_classes = len(le.classes_)
y_onehot = tf.keras.utils.to_categorical(y_encoded, num_classes=num_classes)

# ==================== Scale Features ====================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ==================== Train/Test Split ====================

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_onehot, test_size=0.2, random_state=42)

# ==================== Build Neural Network ====================

model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# ==================== Train Model ====================

history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=16,
    callbacks=[early_stop]
)

# ==================== Evaluate Model ====================

loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy*100:.2f}%")

# ==================== Save Model and Preprocessors ====================

model.save("yoga_pose_model.h5")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(le, "label_encoder.pkl")

print("Model, scaler, and label encoder saved successfully!")

# ==================== Predict Function ====================

def predict_pose_from_angles(angle_row):
    """
    angle_row: 1D array of angles in same order as training
    """
    X_new_scaled = scaler.transform(angle_row.reshape(1, -1))
    y_pred = model.predict(X_new_scaled)
    pose_label = le.inverse_transform([np.argmax(y_pred)])
    return pose_label[0]

# Example usage:
# angle_row = X_test[0]   # pick any row from test set
# print("Predicted Pose:", predict_pose_from_angles(angle_row))
