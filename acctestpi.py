import smbus
import time
import math
import serial
import joblib
import numpy as np

# === Константи ===
I2C_ADDR = 0x53
BUS = smbus.SMBus(1)  # I2C-1 на Raspberry Pi
BAUD = 115200
SERIAL_PORT = "/dev/rfcomm0"  # адаптер COM-порту

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

# === Ініціалізація серійного порту ===
ser = serial.Serial(SERIAL_PORT, BAUD)
time.sleep(2)

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
            ser.write(output.encode())
            print("📤", output.strip())

        time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n❌ Завершення...")
        ser.close()
        break