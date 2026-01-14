#!/usr/bin/env python3
# train_models_rpi.py
# Навчання SGD-моделей (roll/pitch) прямо на Raspberry Pi з ADXL345 (noise-only режим).
# Налаштування задаються змінними нижче (без argparse). Мінімум print; raw лог у CSV; моделі через joblib.

import os
import time
import math
import csv

import numpy as np
import smbus
import joblib

from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# =========================
# НАЛАШТУВАННЯ (редагуйте тут)
# =========================
I2C_ADDR = 0x53
I2C_BUS_ID = 1

TRAIN_SAMPLES = 1200       # кількість відліків для збору (noise-only)
WINDOW_SIZE = 5            # розмір ковзного вікна
SLEEP_SEC = 0.01           # пауза між відліками; 0 = без паузи

OUT_DIR = "models"
FILE_PREFIX = "adxl345_noise"

# SGD налаштування (можете змінювати пізніше, але зафіксуйте для статті)
SGD_MAX_ITER = 2000
SGD_TOL = 1e-3
SGD_ALPHA = 1e-4           # L2 регуляризація
SGD_RANDOM_STATE = 42


# =========================
# ADXL345 (I2C)
# =========================
def setup_adxl345(bus: smbus.SMBus):
    # POWER_CTL (0x2D): measurement mode
    bus.write_byte_data(I2C_ADDR, 0x2D, 0x08)
    # DATA_FORMAT (0x31): full resolution, ±2g
    bus.write_byte_data(I2C_ADDR, 0x31, 0x08)

def read_adxl345(bus: smbus.SMBus):
    data = bus.read_i2c_block_data(I2C_ADDR, 0x32, 6)

    def convert(l, h):
        val = (h << 8) | l
        return val - 65536 if val > 32767 else val

    ax = convert(data[0], data[1]) / 256.0
    ay = convert(data[2], data[3]) / 256.0
    az = convert(data[4], data[5]) / 256.0
    return ax, ay, az

def accel_to_roll_pitch(ax, ay, az):
    roll = math.atan2(ay, az) * 180.0 / math.pi
    pitch = math.atan2(-ax, math.sqrt(ay * ay + az * az)) * 180.0 / math.pi
    return roll, pitch


# =========================
# Датасет (ковзне вікно)
# =========================
def build_window_dataset(series: np.ndarray, window_size: int):
    """
    X_t = [y_{t-window}, ..., y_{t-1}]
    y_t = y_t
    """
    if len(series) <= window_size:
        raise ValueError("Недостатньо даних для заданого WINDOW_SIZE")

    X = []
    y = []
    for i in range(window_size, len(series)):
        X.append(series[i - window_size:i])
        y.append(series[i])
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # I2C init
    bus = smbus.SMBus(I2C_BUS_ID)
    setup_adxl345(bus)

    # Buffers
    t_list = []
    ax_list, ay_list, az_list = [], [], []
    roll_list, pitch_list = [], []

    # Rare progress prints
    step = max(1, TRAIN_SAMPLES // 6)
    next_info = step

    print(f"Start capture: TRAIN_SAMPLES={TRAIN_SAMPLES}, WINDOW_SIZE={WINDOW_SIZE}, SLEEP_SEC={SLEEP_SEC}")
    print("IMPORTANT: sensor must be fixed (noise-only). Ctrl+C to stop early.")

    start = time.perf_counter()

    try:
        for i in range(TRAIN_SAMPLES):
            t = time.perf_counter()
            ax, ay, az = read_adxl345(bus)
            roll, pitch = accel_to_roll_pitch(ax, ay, az)

            t_list.append(t)
            ax_list.append(ax); ay_list.append(ay); az_list.append(az)
            roll_list.append(roll); pitch_list.append(pitch)

            if (i + 1) == next_info:
                print(f"Captured: {i + 1}/{TRAIN_SAMPLES}")
                next_info += step

            if SLEEP_SEC > 0:
                time.sleep(SLEEP_SEC)

    except KeyboardInterrupt:
        print("\nCapture interrupted by user.")

    end = time.perf_counter()

    n = len(roll_list)
    if n <= WINDOW_SIZE + 5:
        raise RuntimeError(f"Too few samples for training: n={n}")

    # Estimate fs from timestamps
    t_arr = np.asarray(t_list, dtype=np.float64)
    dt = np.diff(t_arr)
    dt_med = float(np.median(dt)) if len(dt) else float("nan")
    fs_est = (1.0 / dt_med) if dt_med and not math.isnan(dt_med) else float("nan")

    # Save raw CSV
    csv_path = os.path.join(OUT_DIR, f"{FILE_PREFIX}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "ax", "ay", "az", "roll_raw", "pitch_raw"])
        for row in zip(t_list, ax_list, ay_list, az_list, roll_list, pitch_list):
            w.writerow(row)

    # Build datasets
    roll_arr = np.asarray(roll_list, dtype=np.float32)
    pitch_arr = np.asarray(pitch_list, dtype=np.float32)

    X_roll, y_roll = build_window_dataset(roll_arr, WINDOW_SIZE)
    X_pitch, y_pitch = build_window_dataset(pitch_arr, WINDOW_SIZE)

    # Models: StandardScaler + SGDRegressor
    roll_model = make_pipeline(
        StandardScaler(),
        SGDRegressor(
            loss="squared_error",
            penalty="l2",
            alpha=SGD_ALPHA,
            max_iter=SGD_MAX_ITER,
            tol=SGD_TOL,
            random_state=SGD_RANDOM_STATE
        )
    )

    pitch_model = make_pipeline(
        StandardScaler(),
        SGDRegressor(
            loss="squared_error",
            penalty="l2",
            alpha=SGD_ALPHA,
            max_iter=SGD_MAX_ITER,
            tol=SGD_TOL,
            random_state=SGD_RANDOM_STATE
        )
    )

    print("Training models...")
    roll_model.fit(X_roll, y_roll)
    pitch_model.fit(X_pitch, y_pitch)

    # Save models
    roll_path = os.path.join(OUT_DIR, "roll_model.joblib")
    pitch_path = os.path.join(OUT_DIR, "pitch_model.joblib")

    joblib.dump(roll_model, roll_path)
    joblib.dump(pitch_model, pitch_path)

    duration = end - start

    print("Done.")
    print(f"Saved: {csv_path}")
    print(f"Saved: {roll_path}, {pitch_path}")
    print(f"Captured n={n}, duration={duration:.2f}s, fs~{fs_est:.2f} Hz (median dt={dt_med:.6f}s)")


if __name__ == "__main__":
    main()
