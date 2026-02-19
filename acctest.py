import serial
import time
import joblib
import numpy as np
import threading
import sys
from collections import deque

# === Налаштування ===
INPUT_PORT = 'COM3'
OUTPUT_PORT = 'COM4'
BAUD = 115200
WINDOW_SIZE = 5

# === Завантаження моделей ===
roll_model = joblib.load("roll_model.joblib")
pitch_model = joblib.load("pitch_model.joblib")

# === Буфери для ковзного вікна ===
roll_buffer = deque(maxlen=WINDOW_SIZE)
pitch_buffer = deque(maxlen=WINDOW_SIZE)

# === Відкриття портів ===
ser_in = serial.Serial(INPUT_PORT, BAUD, timeout=1)
ser_out = serial.Serial(OUTPUT_PORT, BAUD)

time.sleep(2)
print("🔌 З'єднано. Очікування даних... (натисни Ctrl+C для виходу)")

# === Ctrl+C завершення ===
def listen_exit():
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n🚪 Завершення...")
        ser_in.close()
        ser_out.close()
        sys.exit()

threading.Thread(target=listen_exit, daemon=True).start()

# === Основний цикл ===
while True:
    try:
        line = ser_in.readline().decode().strip()
        if ',' in line:
            try:
                roll, pitch = map(float, line.split(',')[:2])
                roll_buffer.append(roll)
                pitch_buffer.append(pitch)

                if len(roll_buffer) == WINDOW_SIZE:
                    roll_input = np.array(roll_buffer).reshape(1, -1)
                    pitch_input = np.array(pitch_buffer).reshape(1, -1)

                    roll_filtered = roll_model.predict(roll_input)[0]
                    pitch_filtered = pitch_model.predict(pitch_input)[0]
                else:
                    # Поки буфер не заповнено — повертаємо сирі значення
                    roll_filtered = roll
                    pitch_filtered = pitch

                output_str = f"{roll_filtered:.2f},{pitch_filtered:.2f}\n"
                ser_out.write(output_str.encode())
                print("📤", output_str.strip())

            except ValueError as e:
                print("⚠️ Помилка формату:", e)
                continue
    except Exception as e:
        print("❌", e)
        break
