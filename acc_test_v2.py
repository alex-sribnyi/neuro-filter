#!/usr/bin/env python3
# acc_test_v2.py
# Онлайн-тест: ADXL345 + обчислення roll/pitch + SGD/EMA/Median/Kalman онлайн.
# Логування у JSONL з timestamp в імені файлу.

import os
import time
import math
import json
from datetime import datetime
from collections import deque

import numpy as np
import smbus
import joblib


# =========================
# НАЛАШТУВАННЯ
# =========================
I2C_ADDR = 0x53
I2C_BUS_ID = 1

MODELS_DIR = "models"
RECORDS_DIR = "records"

ROLL_MODEL_PATH = os.path.join(MODELS_DIR, "roll_model_v2.joblib")
PITCH_MODEL_PATH = os.path.join(MODELS_DIR, "pitch_model_v2.joblib")

FILE_PREFIX = "dynamic_test"

RUN_SECONDS = 90
SLEEP_SEC = 0.01
PRINT_EVERY = 400

WINDOW_SIZE_SGD = 5

EMA_ALPHA = 1/3
MEDIAN_WINDOW = 5
KALMAN_R = 1.38e-2
KALMAN_Q = KALMAN_R / 6
KALMAN_P0 = KALMAN_R

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
    bus.write_byte_data(I2C_ADDR, 0x2D, 0x08)
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
# Фільтри
# =========================
class EMA:
    def __init__(self, alpha: float):
        self.alpha = alpha
        self.state = None

    def update(self, x: float) -> float:
        if self.state is None:
            self.state = x
        else:
            self.state = self.alpha * x + (1.0 - self.alpha) * self.state
        return self.state

class MedianCausal:
    def __init__(self, window: int):
        if window % 2 == 0:
            window += 1
        self.buf = deque(maxlen=window)

    def update(self, x: float) -> float:
        self.buf.append(x)
        arr = sorted(self.buf)
        return arr[len(arr) // 2]

class Kalman1D:
    def __init__(self, Q: float, R: float, P0: float = 1.0):
        self.Q = Q
        self.R = R
        self.P = P0
        self.x = None

    def update(self, z: float) -> float:
        if self.x is None:
            self.x = z
            return self.x

        x_pred = self.x
        P_pred = self.P + self.Q

        K = P_pred / (P_pred + self.R)
        self.x = x_pred + K * (z - x_pred)
        self.P = (1.0 - K) * P_pred
        return self.x

def load_pack(path: str) -> dict:
    pack = joblib.load(path)
    if not isinstance(pack, dict):
        raise TypeError(f"Expected dict pack in {path}, got {type(pack)}")
    for k in ("kind", "window_size", "use_anorm", "model"):
        if k not in pack:
            raise KeyError(f"Pack in {path} missing key '{k}'")
    return pack

def build_features(
    roll_buf, pitch_buf, ax_buf, ay_buf, az_buf, an_buf,
    use_anorm: bool
) -> np.ndarray:
    # порядок має збігатися з train: roll, pitch, ax, ay, az, (|a|)
    if use_anorm:
        x = np.concatenate([
            np.asarray(roll_buf, dtype=np.float32),
            np.asarray(pitch_buf, dtype=np.float32),
            np.asarray(ax_buf, dtype=np.float32),
            np.asarray(ay_buf, dtype=np.float32),
            np.asarray(az_buf, dtype=np.float32),
            np.asarray(an_buf, dtype=np.float32),
        ], axis=0)
    else:
        x = np.concatenate([
            np.asarray(roll_buf, dtype=np.float32),
            np.asarray(pitch_buf, dtype=np.float32),
            np.asarray(ax_buf, dtype=np.float32),
            np.asarray(ay_buf, dtype=np.float32),
            np.asarray(az_buf, dtype=np.float32),
        ], axis=0)
    return x.reshape(1, -1)

def main():
    os.makedirs(RECORDS_DIR, exist_ok=True)

    # --- load packs (dict) ---
    roll_pack = load_pack(ROLL_MODEL_PATH)
    pitch_pack = load_pack(PITCH_MODEL_PATH)

    # sanity prints (можна прибрати потім)
    print(type(roll_pack), roll_pack.keys())

    # --- sync params from packs ---
    W = int(roll_pack["window_size"])
    if int(pitch_pack["window_size"]) != W:
        raise ValueError("roll/pitch packs have different window_size")

    use_anorm = bool(roll_pack["use_anorm"])
    if bool(pitch_pack["use_anorm"]) != use_anorm:
        raise ValueError("roll/pitch packs have different use_anorm")

    kind = str(roll_pack["kind"])
    if str(pitch_pack["kind"]) != kind:
        raise ValueError("roll/pitch packs have different kind")

    roll_model = roll_pack["model"]
    pitch_model = pitch_pack["model"]

    # --- i2c ---
    bus = smbus.SMBus(I2C_BUS_ID)
    setup_adxl345(bus)

    log_path = os.path.join(RECORDS_DIR, ts_name(FILE_PREFIX, "jsonl"))

    # --- buffers for expanded features (must match training) ---
    roll_buf  = deque(maxlen=W)
    pitch_buf = deque(maxlen=W)
    ax_buf    = deque(maxlen=W)
    ay_buf    = deque(maxlen=W)
    az_buf    = deque(maxlen=W)
    an_buf    = deque(maxlen=W)  # |a|

    # If kind == "delta", we need integrator states
    roll_state = None
    pitch_state = None

    # --- classic filters ---
    ema_roll = EMA(EMA_ALPHA)
    ema_pitch = EMA(EMA_ALPHA)

    med_roll = MedianCausal(MEDIAN_WINDOW)
    med_pitch = MedianCausal(MEDIAN_WINDOW)

    kal_roll = Kalman1D(KALMAN_Q, KALMAN_R, KALMAN_P0)
    kal_pitch = Kalman1D(KALMAN_Q, KALMAN_R, KALMAN_P0)

    start = time.perf_counter()
    end_time = start + RUN_SECONDS

    print(f"TEST v2: {RUN_SECONDS}s, sleep={SLEEP_SEC}, W={W}, kind={kind}, use_anorm={use_anorm}")
    print(f"Log: {log_path}")

    i = 0
    t_list = []

    with open(log_path, "w", encoding="utf-8") as f:
        try:
            while time.perf_counter() < end_time:
                t = time.perf_counter()

                ax, ay, az = read_adxl345(bus)
                an = math.sqrt(ax*ax + ay*ay + az*az)
                roll, pitch = accel_to_roll_pitch(ax, ay, az)

                # update buffers (must be BEFORE feature build for next step)
                roll_buf.append(roll)
                pitch_buf.append(pitch)
                ax_buf.append(ax)
                ay_buf.append(ay)
                az_buf.append(az)
                an_buf.append(an)

                # --- SGD/ML output ---
                if len(roll_buf) == W:
                    X = build_features(roll_buf, pitch_buf, ax_buf, ay_buf, az_buf, an_buf, use_anorm)

                    if kind == "delta":
                        # model predicts delta; integrate
                        if roll_state is None:
                            roll_state = float(roll_buf[-1])
                            pitch_state = float(pitch_buf[-1])

                        droll = float(roll_model.predict(X)[0])
                        dpitch = float(pitch_model.predict(X)[0])

                        roll_state += droll
                        pitch_state += dpitch

                        roll_sgd = float(roll_state)
                        pitch_sgd = float(pitch_state)
                    else:
                        # model predicts absolute value
                        roll_sgd = float(roll_model.predict(X)[0])
                        pitch_sgd = float(pitch_model.predict(X)[0])
                else:
                    roll_sgd = float(roll)
                    pitch_sgd = float(pitch)

                # --- classic filters on raw ---
                roll_ema = float(ema_roll.update(roll))
                pitch_ema = float(ema_pitch.update(pitch))

                roll_med = float(med_roll.update(roll))
                pitch_med = float(med_pitch.update(pitch))

                roll_kal = float(kal_roll.update(roll))
                pitch_kal = float(kal_pitch.update(pitch))

                jsonl_write(f, {
                    "t": float(t),
                    "ax": float(ax), "ay": float(ay), "az": float(az),
                    "roll_raw": float(roll), "pitch_raw": float(pitch),
                    "roll_sgd": roll_sgd, "pitch_sgd": pitch_sgd,
                    "roll_ema": roll_ema, "pitch_ema": pitch_ema,
                    "roll_med": roll_med, "pitch_med": pitch_med,
                    "roll_kal": roll_kal, "pitch_kal": pitch_kal,
                })

                t_list.append(t)
                i += 1
                if i % PRINT_EVERY == 0:
                    print(f"Samples: {i}")

                if SLEEP_SEC > 0:
                    time.sleep(SLEEP_SEC)

        except KeyboardInterrupt:
            print("Interrupted by user.")

    # fs estimate
    if len(t_list) > 2:
        dts = np.diff(np.array(t_list, dtype=np.float64))
        dt_med = float(np.median(dts))
        fs = 1.0 / dt_med if dt_med > 0 else float("nan")
        print(f"Done. samples={len(t_list)}, fs~{fs:.2f} Hz (median dt={dt_med:.6f}s)")
    else:
        print("Done. Too few samples for fs.")


if __name__ == "__main__":
    main()
