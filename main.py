from machine import I2C, Pin
import utime
import math

# ==== Ініціалізація I2C та ADXL345 ====
i2c = I2C(0, scl=Pin(22), sda=Pin(21), freq=400_000)

def to_signed(val):
    return val - 65536 if val > 32767 else val

class ADXL345:
    def __init__(self, i2c, address=0x53):
        self.i2c = i2c
        self.address = address
        self.i2c.writeto_mem(self.address, 0x2D, b'\x08')  # POWER_CTL: вимірювання
        self.i2c.writeto_mem(self.address, 0x31, b'\x08')  # DATA_FORMAT: full res ±2g

    def read(self):
        data = self.i2c.readfrom_mem(self.address, 0x32, 6)
        x = to_signed(int.from_bytes(data[0:2], 'little', True))
        y = to_signed(int.from_bytes(data[2:4], 'little', True))
        z = to_signed(int.from_bytes(data[4:6], 'little', True))
        return x / 256.0, y / 256.0, z / 256.0

accel = ADXL345(i2c)

while True:
    xg, yg, zg = accel.read()
    roll = math.atan2(yg, zg) * 180 / math.pi
    pitch = math.atan2(-xg, math.sqrt(yg ** 2 + zg ** 2)) * 180 / math.pi

    print(f"{roll:.2f},{pitch:.2f}")
    utime.sleep(0.01)
