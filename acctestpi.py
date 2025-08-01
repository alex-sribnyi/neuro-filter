import smbus
import socket
import time
import math
import joblib
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

# === Параметри графіків ===
PLOT_WINDOW = 100
roll_raw_buffer = deque(maxlen=PLOT_WINDOW)
pitch_raw_buffer = deque(maxlen=PLOT_WINDOW)
roll_filt_buffer = deque(maxlen=PLOT_WINDOW)
pitch_filt_buffer = deque(maxlen=PLOT_WINDOW)

# === Підготовка графіків ===
plt.ion()
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))

line1_raw, = ax1.plot([], [], label="Raw Roll", color='orange')
line1_filt, = ax1.plot([], [], label="Filtered Roll", color='blue')
ax1.set_ylim(-90, 90)
ax1.set_title("Roll")
ax1.legend()

line2_raw, = ax2.plot([], [], label="Raw Pitch", color='orange')
line2_filt, = ax2.plot([], [], label="Filtered Pitch", color='blue')
ax2.set_ylim(-90, 90)
ax2.set_title("Pitch")
ax2.legend()

def update_plot(roll, pitch, roll_filt, pitch_filt):
    roll_raw_buffer.append(roll)
    pitch_raw_buffer.append(pitch)
    roll_filt_buffer.append(roll_filt)
    pitch_filt_buffer.append(pitch_filt)

    x = np.arange(len(roll_raw_buffer))

    line1_raw.set_data(x, roll_raw_buffer)
    line1_filt.set_data(x, roll_filt_buffer)
    ax1.set_xlim(0, len(roll_raw_buffer))

    line2_raw.set_data(x, pitch_raw_buffer)
    line2_filt.set_data(x, pitch_filt_buffer)
    ax2.set_xlim(0, len(pitch_raw_buffer))

    fig.canvas.draw()
    fig.canvas.flush_events()

# === Константи ===
I2C_ADDR = 0x53
BUS = smbus.SMBus(1)  # I2C-1 на Raspberry Pi
HOST = '192.168.0.2'
PORT = 5000

WINDOW_SIZE = 5
roll_buffer = []
pitch_buffer = []

# === Ініціалізація акселерометра ===
def setup_adxl345():
    BUS.write_byte_data(I2C_ADDR, 0x2D, 0x08)  # POWER_CTL: вимірювання
    BUS.write_byte_data(I2C_ADDR, 0x31, 0x08)  # DATA_FORMAT: full res ±2g

def read_adxl345():
    data = BUS.read_i2c_block_data(I2C_ADDR, 0x32, 6)
    def convert(l, h):  # little endian signed
        val = h << 8 | l
        return val - 65536 if val > 32767 else val

    x = convert(data[0], data[1]) / 256.0
    y = convert(data[2], data[3]) / 256.0
    z = convert(data[4], data[5]) / 256.0
    return x, y, z

# === Ініціалізація моделей ===
roll_model = joblib.load("roll_model.joblib")
pitch_model = joblib.load("pitch_model.joblib")

# === Ініціалізація ===
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.bind((HOST, PORT))
sock.listen(1)
conn, addr = sock.accept()

# === Основний цикл ===
setup_adxl345()
print("📡 Старт фільтрації та передачі...")

while True:
    try:
        xg, yg, zg = read_adxl345()
        roll = math.atan2(yg, zg) * 180 / math.pi
        pitch = math.atan2(-xg, math.sqrt(yg**2 + zg**2)) * 180 / math.pi

        roll_buffer.append(roll)
        pitch_buffer.append(pitch)

        if len(roll_buffer) >= WINDOW_SIZE:
            X_roll = np.array(roll_buffer[-WINDOW_SIZE:]).reshape(1, -1)
            X_pitch = np.array(pitch_buffer[-WINDOW_SIZE:]).reshape(1, -1)

            roll_filt = roll_model.predict(X_roll)[0]
            pitch_filt = pitch_model.predict(X_pitch)[0]

            output = f"{roll_filt:.2f},{pitch_filt:.2f}\n"
            conn.sendall(output.encode())
            time.sleep(0.01)
            print("📤", output.strip())
            update_plot(roll, pitch, roll_filt, pitch_filt)
        time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n❌ Завершення...")
        conn.close()
        break