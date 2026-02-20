#!/usr/bin/env python3
# acc_train_v2.py
# Навчання ML-фільтра (SGD) на Raspberry Pi з ADXL345 у форматі "статичні сегменти + пауза на зміну пози".
# ВАЖЛИВІ зміни для стабільності SGD:
#  1) Формування вікон НЕ перетинає межі segment_id
#  2) Масштабування ознак (StandardScaler)
#  3) Контроль кроку SGD (learning_rate="constant", eta0=1e-3) + слабка L2-регуляризація (alpha=1e-6)
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

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor


# =========================
# НАЛАШТУВАННЯ
# =========================
I2C_ADDR = 0x53
I2C_BUS_ID = 1

SEGMENTS = 10
HOLD_SECONDS = 12.0
SLEEP_SEC = 0.01

ZERO_EPS = 1e-9
MAX_BAD_STREAK = 25     # скільки підряд "поганих" семплів допустимо

WINDOW_SIZE = 9
WARMUP_SECONDS = 0.7

MODELS_DIR = "models"
RECORDS_DIR = "records"

ROLL_MODEL_PATH = os.path.join(MODELS_DIR, "roll_model_v2.joblib")
PITCH_MODEL_PATH = os.path.join(MODELS_DIR, "pitch_model_v2.joblib")

# SGD стабільні параметри
SGD_ALPHA = 1e-6
SGD_ETA0 = 1e-3
SGD_MAX_ITER = 4000
SGD_TOL = 1e-4
SGD_RANDOM_STATE = 42


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

    for i in range(window, n):
        seg = seg_ids[i]
        # вікно [i-window, ..., i] має бути всередині одного segment_id
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
    seg_all = []

    print(f"TRAIN v2 (segmented): segments={SEGMENTS}, hold={HOLD_SECONDS}s, warmup={WARMUP_SECONDS}s, sleep={SLEEP_SEC}s")
    print("Workflow: set position -> press Enter -> wait -> capture -> pause -> change position -> press Enter ...")
    print(f"Log: {log_path}")

    bad_streak = 0

    with open(log_path, "w", encoding="utf-8") as f:
        for seg in range(1, SEGMENTS + 1):
            input(f"\n[{seg}/{SEGMENTS}] Set new static position and press Enter to START capture...")

            # Warmup (затухання)
            warmup_end = time.perf_counter() + WARMUP_SECONDS
            while time.perf_counter() < warmup_end:
                ax, ay, az = read_adxl345(bus)
                if abs(ax) < ZERO_EPS and abs(ay) < ZERO_EPS and abs(az) < ZERO_EPS:
                    bad_streak += 1
                    if bad_streak >= MAX_BAD_STREAK:
                        print("ERROR: ADXL345 returns zeros repeatedly. Reinitializing sensor...")
                        setup_adxl345(bus)
                        bad_streak = 0
                    continue
                else:
                    bad_streak = 0
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
                seg_all.append(seg)

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

    # Вікна без перетину сегментів
    X_roll, y_roll = build_window_dataset_segmented(roll_arr, seg_arr, WINDOW_SIZE)
    X_pitch, y_pitch = build_window_dataset_segmented(pitch_arr, seg_arr, WINDOW_SIZE)

    print("\nTraining models...")

    # Масштабування ознак + стабільний SGD
    roll_model = make_pipeline(
        StandardScaler(),
        SGDRegressor(
            loss="squared_error",
            penalty="l2",
            alpha=SGD_ALPHA,
            learning_rate="constant",
            eta0=SGD_ETA0,
            max_iter=SGD_MAX_ITER,
            tol=SGD_TOL,
            shuffle=True,
            random_state=SGD_RANDOM_STATE
        )
    )

    pitch_model = make_pipeline(
        StandardScaler(),
        SGDRegressor(
            loss="squared_error",
            penalty="l2",
            alpha=SGD_ALPHA,
            learning_rate="constant",
            eta0=SGD_ETA0,
            max_iter=SGD_MAX_ITER,
            tol=SGD_TOL,
            shuffle=True,
            random_state=SGD_RANDOM_STATE
        )
    )

    roll_model.fit(X_roll, y_roll)
    pitch_model.fit(X_pitch, y_pitch)

    joblib.dump(roll_model, ROLL_MODEL_PATH)
    joblib.dump(pitch_model, PITCH_MODEL_PATH)

    # sanity-check масштабу коефіцієнтів
    w_roll = roll_model.named_steps["sgdregressor"].coef_
    b_roll = roll_model.named_steps["sgdregressor"].intercept_[0]
    w_pitch = pitch_model.named_steps["sgdregressor"].coef_
    b_pitch = pitch_model.named_steps["sgdregressor"].intercept_[0]

    print("Done.")
    print(f"Samples total: {n}, usable for train: roll={len(X_roll)}, pitch={len(X_pitch)}")
    print(f"fs~{fs:.2f} Hz (median dt={dt_med:.6f}s)")
    print(f"Models: {ROLL_MODEL_PATH}, {PITCH_MODEL_PATH}")
    print(f"Train log: {log_path}")
    print(f"Sanity |roll coef|max={float(np.max(np.abs(w_roll))):.6f}, intercept={float(b_roll):.6f}")
    print(f"Sanity |pitch coef|max={float(np.max(np.abs(w_pitch))):.6f}, intercept={float(b_pitch):.6f}")


if __name__ == "__main__":
    main()
