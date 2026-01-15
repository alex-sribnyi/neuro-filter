#!/usr/bin/env python3
# acc_train_v2.py
# Навчання ML-фільтра (SGD) на Raspberry Pi з ADXL345 у форматі "статичні сегменти + пауза на зміну пози".
#
# ВАРІАНТ A (без teacher/EMA-trend):
#   - Навчаємо не y[k], а приріст Δy[k] = y[k] - y[k-1]
#   - У runtime: y_hat[k] = y_hat[k-1] + Δy_hat[k]
#
# Додатково:
#   - Розширені ознаки: історія roll, pitch, ax, ay, az, |a| у ковзному вікні
#   - Вікна НЕ перетинають межі segment_id
#
# Вихід:
#   models/roll_model_v2.joblib
#   models/pitch_model_v2.joblib
#   records/noise_train_YYYY-MM-DD_HH-MM-SS.jsonl (лог навчання)

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

WINDOW_SIZE = 5
WARMUP_SECONDS = 0.7

MODELS_DIR = "models"
RECORDS_DIR = "records"

ROLL_MODEL_PATH = os.path.join(MODELS_DIR, "roll_model_v2.joblib")
PITCH_MODEL_PATH = os.path.join(MODELS_DIR, "pitch_model_v2.joblib")

# SGD (стабільні параметри + StandardScaler)
SGD_ALPHA = 1e-6
SGD_ETA0 = 1e-3
SGD_MAX_ITER = 8000
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

def estimate_fs(t_list):
    if len(t_list) < 3:
        return float("nan"), float("nan")
    dts = np.diff(np.array(t_list, dtype=np.float64))
    dt_med = float(np.median(dts))
    fs = 1.0 / dt_med if dt_med > 0 else float("nan")
    return fs, dt_med


# =========================
# ADXL345
# =========================
def setup_adxl345(bus: smbus.SMBus):
    bus.write_byte_data(I2C_ADDR, 0x2D, 0x08)  # POWER_CTL: measurement
    bus.write_byte_data(I2C_ADDR, 0x31, 0x08)  # DATA_FORMAT: full res ±2g

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
# Dataset
# =========================
def build_delta_dataset_segmented(
    roll_raw: np.ndarray,
    pitch_raw: np.ndarray,
    ax: np.ndarray, ay: np.ndarray, az: np.ndarray,
    seg_ids: np.ndarray,
    window: int,
    use_anorm: bool = True
):
    """
    X_t = [roll_{t-window}..roll_{t-1},
           pitch_{t-window}..pitch_{t-1},
           ax_{t-window}..ax_{t-1},
           ay_{...}, az_{...}, |a|_{...}]
    y_t = Δroll_t  = roll_t  - roll_{t-1}
    z_t = Δpitch_t = pitch_t - pitch_{t-1}

    Умова: індекси [t-window, ..., t] належать одному segment_id.
    """
    n = len(roll_raw)
    if not (n == len(pitch_raw) == len(ax) == len(ay) == len(az) == len(seg_ids)):
        raise ValueError("Усі масиви повинні мати однакову довжину")

    if n < window + 2:
        raise RuntimeError("Занадто мало даних для формування вікон")

    if use_anorm:
        anorm = np.sqrt(ax * ax + ay * ay + az * az).astype(np.float32)

    X, y_droll, y_dpitch = [], [], []

    for t in range(window, n):
        seg = seg_ids[t]
        if not np.all(seg_ids[t - window:t + 1] == seg):
            continue

        r_hist = roll_raw[t - window:t]
        p_hist = pitch_raw[t - window:t]
        ax_hist = ax[t - window:t]
        ay_hist = ay[t - window:t]
        az_hist = az[t - window:t]

        if use_anorm:
            a_hist = anorm[t - window:t]
            feat = np.concatenate([r_hist, p_hist, ax_hist, ay_hist, az_hist, a_hist], axis=0)
        else:
            feat = np.concatenate([r_hist, p_hist, ax_hist, ay_hist, az_hist], axis=0)

        droll = roll_raw[t] - roll_raw[t - 1]
        dpitch = pitch_raw[t] - pitch_raw[t - 1]

        X.append(feat)
        y_droll.append(droll)
        y_dpitch.append(dpitch)

    X = np.asarray(X, dtype=np.float32)
    y_droll = np.asarray(y_droll, dtype=np.float32)
    y_dpitch = np.asarray(y_dpitch, dtype=np.float32)

    if len(X) == 0:
        raise RuntimeError("Після сегментації не залишилось даних для навчання. Збільшіть HOLD_SECONDS або зменшіть WINDOW_SIZE.")

    return X, y_droll, y_dpitch


def make_sgd_pipeline():
    return make_pipeline(
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
            random_state=SGD_RANDOM_STATE,
        )
    )


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RECORDS_DIR, exist_ok=True)

    bus = smbus.SMBus(I2C_BUS_ID)
    setup_adxl345(bus)

    log_path = os.path.join(RECORDS_DIR, ts_name("noise_train", "jsonl"))

    t_all = []
    roll_all, pitch_all = [], []
    ax_all, ay_all, az_all = [], [], []
    seg_all = []

    print(f"TRAIN v2 (delta): segments={SEGMENTS}, hold={HOLD_SECONDS}s, warmup={WARMUP_SECONDS}s, sleep={SLEEP_SEC}s")
    print(f"Log: {log_path}")

    with open(log_path, "w", encoding="utf-8") as f:
        for seg in range(1, SEGMENTS + 1):
            input(f"\n[{seg}/{SEGMENTS}] Set new static position and press Enter to START capture...")

            warmup_end = time.perf_counter() + WARMUP_SECONDS
            while time.perf_counter() < warmup_end:
                _ = read_adxl345(bus)
                if SLEEP_SEC > 0:
                    time.sleep(SLEEP_SEC)

            seg_end = time.perf_counter() + HOLD_SECONDS

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
                roll_all.append(roll); pitch_all.append(pitch)
                ax_all.append(ax); ay_all.append(ay); az_all.append(az)
                seg_all.append(seg)

                if SLEEP_SEC > 0:
                    time.sleep(SLEEP_SEC)

            print(f"Segment {seg} captured.")

    n = len(roll_all)
    if n <= WINDOW_SIZE + 5:
        raise RuntimeError(f"Too few samples for training: n={n}")

    fs, dt_med = estimate_fs(t_all)

    roll_raw = np.asarray(roll_all, dtype=np.float32)
    pitch_raw = np.asarray(pitch_all, dtype=np.float32)
    ax = np.asarray(ax_all, dtype=np.float32)
    ay = np.asarray(ay_all, dtype=np.float32)
    az = np.asarray(az_all, dtype=np.float32)
    seg_ids = np.asarray(seg_all, dtype=np.int32)

    X, y_droll, y_dpitch = build_delta_dataset_segmented(
        roll_raw=roll_raw,
        pitch_raw=pitch_raw,
        ax=ax, ay=ay, az=az,
        seg_ids=seg_ids,
        window=WINDOW_SIZE,
        use_anorm=True,
    )

    print(f"Usable windows: {len(X)} / total samples: {n}; fs~{fs:.2f} Hz (median dt={dt_med:.6f}s)")

    roll_model = make_sgd_pipeline()
    pitch_model = make_sgd_pipeline()

    roll_model.fit(X, y_droll)
    pitch_model.fit(X, y_dpitch)

    # Зберігаємо як dict з метаданими: test-скрипт має знати, що це Δ-модель і які ознаки подавати.
    roll_pack = {
        "kind": "delta",
        "target": "droll",
        "window_size": WINDOW_SIZE,
        "features": ["roll_hist", "pitch_hist", "ax_hist", "ay_hist", "az_hist", "anorm_hist"],
        "use_anorm": True,
        "model": roll_model,
    }
    pitch_pack = {
        "kind": "delta",
        "target": "dpitch",
        "window_size": WINDOW_SIZE,
        "features": ["roll_hist", "pitch_hist", "ax_hist", "ay_hist", "az_hist", "anorm_hist"],
        "use_anorm": True,
        "model": pitch_model,
    }

    joblib.dump(roll_pack, ROLL_MODEL_PATH)
    joblib.dump(pitch_pack, PITCH_MODEL_PATH)

    # sanity-check (коротко)
    w_r = roll_model.named_steps["sgdregressor"].coef_
    w_p = pitch_model.named_steps["sgdregressor"].coef_
    print(f"Saved models: {ROLL_MODEL_PATH}, {PITCH_MODEL_PATH}")
    print(f"Sanity |coef|max: roll={float(np.max(np.abs(w_r))):.6f}, pitch={float(np.max(np.abs(w_p))):.6f}")
    print(f"Train log: {log_path}")


if __name__ == "__main__":
    main()
