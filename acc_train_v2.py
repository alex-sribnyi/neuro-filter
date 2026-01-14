#!/usr/bin/env python3
# acc_train_v2.py
# Навчання ML-фільтра (SGD) на Raspberry Pi з ADXL345 у форматі "статичні сегменти + пауза на зміну пози".
# Ключове виправлення: формування ковзних вікон НЕ ПОВИННО перетинати межі segment_id.
#
# Вихід:
#   models/roll_model_v2.joblib
#   models/pitch_model_v2.joblib
#   records/noise_train_YYYY-MM-DD_HH-MM-SS.jsonl

import os
import time
import math
import json
from datetime import datetime

import numpy as np
import smbus
import joblib

from sklearn.linear_model import SGDRegressor


# =========================
# НАЛАШТУВАННЯ
# =========================
I2C_ADDR = 0x53
I2C_BUS_ID = 1

SEGMENTS = 10
HOLD_SECONDS = 12.0
SLEEP_SEC = 0.01

WINDOW_SIZE = 5
WARMUP_SECONDS = 0.7

MODELS_DIR = "models"
RECORDS_DIR = "records"

ROLL_MODEL_PATH = os.path.join(MODELS_DIR, "roll_model_v2.joblib")
PITCH_MODEL_PATH = os.path.join(MODELS_DIR, "pitch_model_v2.joblib")

SGD_MAX_ITER = 1000
SGD_TOL = 1e-3


# =========================
# Utilities
# =========================
def ts_name(prefix: str, ext: str) -> str:
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return f"{prefix}_{stamp}.{ext}"

def jsonl_write(fp, obj):
    fp.write(json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n")


# =========================
# ADXL345
# =========================
def setup_adxl345(bus: smbus.SMBus):
    bus.write_byte_data(I2C_ADDR, 0x2D, 0x08)  # POWER_CTL measurement
    bus.write_byte_data(I2C_ADDR, 0x31, 0x08)  # DATA_FORMAT full res ±2g

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
    pitch = math.atan2(-ax, math.sqrt(ay**2 + az**2)) * 180.0 / math.pi
    return roll, pitch


# =========================
# Dataset
# =========================
def estimate_fs(t_list):
    if len(t_list) < 3:
        return float("nan"), float("nan")
    dts = np.diff(np.array(t_list, dtype=np.float64))
    dt_med = float(np.median(dts))
    fs = 1.0 / dt_med if dt_med > 0 else float("nan")
    return fs, dt_med

def build_window_dataset_segmented(series: np.ndarray, seg_ids: np.ndarray, window: int):
    """
    Формує X/y так, щоб жодне ковзне вікно НЕ перетинало межі segment_id.
    """
    X, y = [], []
    n = len(series)
    if n != len(seg_ids):
        raise ValueError("series і seg_ids повинні мати однакову довжину")

    # проходимо послідовно; для кожної точки i перевіряємо, що seg_ids на [i-window, i] однаковий
    for i in range(window, n):
        seg = seg_ids[i]
        if np.all(seg_ids[i - window:i + 1] == seg):
            X.append(series[i - window:i])
            y.append(series[i])

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)

    if len(X) == 0:
        raise RuntimeError("Після сегментації не залишилось даних для навчання. Збільшіть HOLD_SECONDS або зменшіть WINDOW_SIZE.")

    return X, y


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RECORDS_DIR, exist_ok=True)

    bus = smbus.SMBus(I2C_BUS_ID)
    setup_adxl345(bus)

    log_path = os.path.join(RECORDS_DIR, ts_name("noise_train", "jsonl"))

    t_all = []
    roll_all = []
    pitch_all = []
    seg_all = []  # <-- ДОДАНО: зберігаємо segment_id для кожного відліку

    print(f"TRAIN v2 (segmented): segments={SEGMENTS}, hold={HOLD_SECONDS}s, warmup={WARMUP_SECONDS}s, sleep={SLEEP_SEC}s")
    print("Workflow: set position -> press Enter -> wait -> capture -> pause -> change position -> press Enter ...")
    print(f"Log: {log_path}")

    with open(log_path, "w", encoding="utf-8") as f:
        for seg in range(1, SEGMENTS + 1):
            input(f"\n[{seg}/{SEGMENTS}] Set new static position and press Enter to START capture...")

            # Warmup (затухання)
            warmup_end = time.perf_counter() + WARMUP_SECONDS
            while time.perf_counter() < warmup_end:
                _ = read_adxl345(bus)
                if SLEEP_SEC > 0:
                    time.sleep(SLEEP_SEC)

            # Capture segment
            seg_start = time.perf_counter()
            seg_end = seg_start + HOLD_SECONDS

            while time.perf_counter() < seg_end:
                t = time.perf_counter()
                ax, ay, az = read_adxl345(bus)
                roll, pitch = accel_to_roll_pitch(ax, ay, az)

                jsonl_write(f, {
                    "t": float(t),
                    "segment_id": int(seg),
                    "ax": float(ax), "ay": float(ay), "az": float(az),
                    "roll_raw": float(roll),
                    "pitch_raw": float(pitch),
                })

                t_all.append(t)
                roll_all.append(roll)
                pitch_all.append(pitch)
                seg_all.append(seg)  # <-- ДОДАНО

                if SLEEP_SEC > 0:
                    time.sleep(SLEEP_SEC)

            print(f"Segment {seg} captured.")

    n = len(roll_all)
    if n <= WINDOW_SIZE + 5:
        raise RuntimeError(f"Too few samples for training: n={n}")

    fs, dt_med = estimate_fs(t_all)

    roll_arr = np.asarray(roll_all, dtype=np.float32)
    pitch_arr = np.asarray(pitch_all, dtype=np.float32)
    seg_arr = np.asarray(seg_all, dtype=np.int32)

    # ВИПРАВЛЕННЯ: формуємо датасет без перетину меж сегментів
    X_roll, y_roll = build_window_dataset_segmented(roll_arr, seg_arr, WINDOW_SIZE)
    X_pitch, y_pitch = build_window_dataset_segmented(pitch_arr, seg_arr, WINDOW_SIZE)

    print("\nTraining models...")
    roll_model = SGDRegressor(max_iter=SGD_MAX_ITER, tol=SGD_TOL)
    pitch_model = SGDRegressor(max_iter=SGD_MAX_ITER, tol=SGD_TOL)

    roll_model.fit(X_roll, y_roll)
    pitch_model.fit(X_pitch, y_pitch)

    joblib.dump(roll_model, ROLL_MODEL_PATH)
    joblib.dump(pitch_model, PITCH_MODEL_PATH)

    print("Done.")
    print(f"Samples total: {n}, usable for train: roll={len(X_roll)}, pitch={len(X_pitch)}")
    print(f"fs~{fs:.2f} Hz (median dt={dt_med:.6f}s)")
    print(f"Models: {ROLL_MODEL_PATH}, {PITCH_MODEL_PATH}")
    print(f"Train log: {log_path}")


if __name__ == "__main__":
    main()
