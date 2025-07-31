import serial
import time
import numpy as np
from sklearn.linear_model import SGDRegressor
import joblib
import matplotlib.pyplot as plt

PORT = 'COM10'
BAUD = 115200
SAMPLES = 1000
K = 0.05  # EMA коефіцієнт

ser = serial.Serial(PORT, BAUD, timeout=1)
time.sleep(2)

roll_raw = []
pitch_raw = []
roll_ema = []
pitch_ema = []

print("Збір даних...")

rollF = 0
pitchF = 0

while len(roll_raw) < SAMPLES:
    line = ser.readline().decode().strip()
    if ',' in line:
        try:
            roll, pitch = map(float, line.split(',')[:2])
            rollF = K * roll + (1 - K) * rollF
            pitchF = K * pitch + (1 - K) * pitchF

            roll_raw.append([roll])
            roll_ema.append(rollF)

            pitch_raw.append([pitch])
            pitch_ema.append(pitchF)

            print(f"#{len(roll_raw)}: Roll={roll:.2f} → {rollF:.2f}")
        except:
            continue

ser.close()

# Навчання
roll_model = SGDRegressor()
pitch_model = SGDRegressor()

roll_model.fit(roll_raw, roll_ema)
pitch_model.fit(pitch_raw, pitch_ema)

joblib.dump(roll_model, "roll_model.joblib")
joblib.dump(pitch_model, "pitch_model.joblib")

print("Моделі збережено.")

# Візуалізація
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.title("Roll")
plt.plot([r[0] for r in roll_raw], label="Raw")
plt.plot(roll_ema, label="EMA", linestyle='--')
plt.plot(roll_model.predict(roll_raw), label="Predicted", linestyle=':')
plt.legend()

plt.subplot(2, 1, 2)
plt.title("Pitch")
plt.plot([p[0] for p in pitch_raw], label="Raw")
plt.plot(pitch_ema, label="EMA", linestyle='--')
plt.plot(pitch_model.predict(pitch_raw), label="Predicted", linestyle=':')
plt.legend()

plt.tight_layout()
plt.show()
