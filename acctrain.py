import serial
import time
import numpy as np
from sklearn.linear_model import SGDRegressor
import joblib
import matplotlib.pyplot as plt

# === Налаштування ===
PORT = 'COM10'
BAUD = 115200
TRAIN_SAMPLES = 1000
WINDOW_SIZE = 5  # Розмір ковзного вікна

# === Збір даних ===
ser = serial.Serial(PORT, BAUD, timeout=1)
time.sleep(2)

roll_data = []
pitch_data = []

print("⏳ Збір даних...")

while len(roll_data) < TRAIN_SAMPLES:
    line = ser.readline().decode().strip()
    if ',' in line:
        try:
            roll, pitch = map(float, line.split(',')[:2])
            roll_data.append(roll)
            pitch_data.append(pitch)
            print(f"[{len(roll_data)}] Roll: {roll:.2f}, Pitch: {pitch:.2f}")
        except ValueError:
            continue

ser.close()
print("✅ Отримано всі дані.")

# === Побудова навчальних вибірок ===
X_roll = []
y_roll = []
X_pitch = []
y_pitch = []

for i in range(WINDOW_SIZE, len(roll_data)):
    X_roll.append(roll_data[i - WINDOW_SIZE:i])
    y_roll.append(roll_data[i])
    X_pitch.append(pitch_data[i - WINDOW_SIZE:i])
    y_pitch.append(pitch_data[i])

X_roll = np.array(X_roll)
y_roll = np.array(y_roll)
X_pitch = np.array(X_pitch)
y_pitch = np.array(y_pitch)

# === Навчання ===
print("🤖 Навчання моделі...")

roll_model = SGDRegressor(max_iter=1000, tol=1e-3)
pitch_model = SGDRegressor(max_iter=1000, tol=1e-3)

roll_model.fit(X_roll, y_roll)
pitch_model.fit(X_pitch, y_pitch)

print("📊 Roll: w =", roll_model.coef_, ", b =", roll_model.intercept_[0])
print("📊 Pitch: w =", pitch_model.coef_, ", b =", pitch_model.intercept_[0])

# === Збереження моделей ===
joblib.dump(roll_model, "roll_model.joblib")
joblib.dump(pitch_model, "pitch_model.joblib")
print("💾 Моделі збережено.")

# === Візуалізація ===
roll_pred = roll_model.predict(X_roll)
pitch_pred = pitch_model.predict(X_pitch)

plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(range(WINDOW_SIZE, TRAIN_SAMPLES), roll_data[WINDOW_SIZE:], label="Raw Roll")
plt.plot(range(WINDOW_SIZE, TRAIN_SAMPLES), roll_pred, '--', label="Predicted Roll")
plt.title("Roll")
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(range(WINDOW_SIZE, TRAIN_SAMPLES), pitch_data[WINDOW_SIZE:], label="Raw Pitch")
plt.plot(range(WINDOW_SIZE, TRAIN_SAMPLES), pitch_pred, '--', label="Predicted Pitch")
plt.title("Pitch")
plt.legend()

plt.tight_layout()
plt.show()
