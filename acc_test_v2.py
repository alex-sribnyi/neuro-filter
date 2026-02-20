#!/usr/bin/env python3
# acc_test_v2.py
# Онлайн-тест: ADXL345 + roll/pitch + SGD(ML)/EMA/Median/Kalman онлайн.
# Моделі: Pipeline(StandardScaler + SGDRegressor).
# Логування у JSONL з timestamp у назві файлу.

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

# fallback, якщо не зможемо визначити W з моделі
WINDOW_SIZE_FALLBACK = 9

# Фіксовані параметри фільтрів під W=9
EMA_ALPHA = 0.2
MEDIAN_WINDOW = 9

KALMAN_R = 1.38e-2
KALMAN_Q = KALMAN_R / 20
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
    pitch = math.atan2(-ax, math.sqrt(ay * ay + az * az)) * 180.0 / math.pi
    return roll, pitch


# =========================
# Фільтри
# =========================
class EMA:
    def __init__(self, alpha: float):
        self.alpha = float(alpha)
        self.state = None

    def update(self, x: float) -> float:
        x = float(x)
        if self.state is None:
            self.state = x
        else:
            self.state = self.alpha * x + (1.0 - self.alpha) * self.state
        return float(self.state)

class MedianCausal:
    def __init__(self, window: int):
        window = int(window)
        if window % 2 == 0:
            window += 1
        self.buf = deque(maxlen=window)

    def update(self, x: float) -> float:
        self.buf.append(float(x))
        arr = sorted(self.buf)
        return float(arr[len(arr) // 2])

class Kalman1D:
    # 1D KF: x_k = x_{k-1} + w, z_k = x_k + v
    def __init__(self, Q: float, R: float, P0: float):
        self.Q = float(Q)
        self.R = float(R)
        self.P = float(P0)
        self.x = None

    def update(self, z: float) -> float:
        z = float(z)
        if self.x is None:
            self.x = z
            return float(self.x)

        x_pred = self.x
        P_pred = self.P + self.Q

        K = P_pred / (P_pred + self.R)
        self.x = x_pred + K * (z - x_pred)
        self.P = (1.0 - K) * P_pred
        return float(self.x)


# =========================
# Model helpers
# =========================
def infer_expected_features(pipeline) -> int | None:
    """
    Для Pipeline(StandardScaler -> SGDRegressor) дістанемо очікувану кількість ознак.
    Працює якщо StandardScaler вже fitted.
    """
    # try sklearn pipeline API
    scaler = None
    if hasattr(pipeline, "named_steps"):
        scaler = pipeline.named_steps.get("standardscaler", None)

    if scaler is None:
        return None

    # у fitted scaler є n_features_in_
    n = getattr(scaler, "n_features_in_", None)
    if n is None:
        # fallback: mean_ length (також лише після fit)
        mean_ = getattr(scaler, "mean_", None)
        if mean_ is not None:
            return int(len(mean_))
        return None
    return int(n)

def infer_window_from_nfeatures(n_features: int) -> int:
    """
    У вашому train: X має форму [W] (тільки roll або pitch).
    Тобто n_features == W.
    """
    return int(n_features)


def main():
    os.makedirs(RECORDS_DIR, exist_ok=True)

    # --- load models (Pipeline) ---
    roll_model = joblib.load(ROLL_MODEL_PATH)
    pitch_model = joblib.load(PITCH_MODEL_PATH)

    # --- infer W from model (safer than hardcoding) ---
    nfr = infer_expected_features(roll_model)
    nfp = infer_expected_features(pitch_model)

    if nfr is None or nfp is None:
        W = WINDOW_SIZE_FALLBACK
    else:
        if nfr != nfp:
            raise ValueError(f"roll/pitch models expect different n_features: roll={nfr}, pitch={nfp}")
        W = infer_window_from_nfeatures(nfr)

    # sanity: W must be odd? not required; just positive
    if W <= 0:
        raise RuntimeError(f"Invalid inferred window size W={W}")

    # --- i2c ---
    bus = smbus.SMBus(I2C_BUS_ID)
    setup_adxl345(bus)

    log_path = os.path.join(RECORDS_DIR, ts_name(FILE_PREFIX, "jsonl"))

    # --- buffers for SGD ---
    roll_buf = deque(maxlen=W)
    pitch_buf = deque(maxlen=W)

    # --- classic filters ---
    ema_roll = EMA(EMA_ALPHA)
    ema_pitch = EMA(EMA_ALPHA)

    med_roll = MedianCausal(MEDIAN_WINDOW)
    med_pitch = MedianCausal(MEDIAN_WINDOW)

    kal_roll = Kalman1D(KALMAN_Q, KALMAN_R, KALMAN_P0)
    kal_pitch = Kalman1D(KALMAN_Q, KALMAN_R, KALMAN_P0)

    start = time.perf_counter()
    end_time = start + RUN_SECONDS

    print(f"TEST v2: {RUN_SECONDS}s, sleep={SLEEP_SEC}, W(from model)={W}")
    print(f"Filters: EMA_ALPHA={EMA_ALPHA}, MEDIAN_WINDOW={MEDIAN_WINDOW}, "
          f"KF(Q={KALMAN_Q:.3e}, R={KALMAN_R:.3e}, P0={KALMAN_P0:.3e})")
    print(f"Log: {log_path}")

    i = 0
    t_list = []

    with open(log_path, "w", encoding="utf-8") as f:
        try:
            while time.perf_counter() < end_time:
                t = time.perf_counter()

                ax, ay, az = read_adxl345(bus)
                roll, pitch = accel_to_roll_pitch(ax, ay, az)

                # update buffers
                roll_buf.append(roll)
                pitch_buf.append(pitch)

                # SGD output (поки буфер не заповнений — raw)
                if len(roll_buf) == W:
                    Xr = np.array(roll_buf, dtype=np.float32).reshape(1, -1)
                    Xp = np.array(pitch_buf, dtype=np.float32).reshape(1, -1)

                    # якщо модель очікує іншу кількість ознак — впадемо тут,
                    # але тепер це буде через неправильні моделі, а не код
                    roll_sgd = float(roll_model.predict(Xr)[0])
                    pitch_sgd = float(pitch_model.predict(Xp)[0])
                else:
                    roll_sgd = float(roll)
                    pitch_sgd = float(pitch)

                # classic filters on raw
                roll_ema = ema_roll.update(roll)
                pitch_ema = ema_pitch.update(pitch)

                roll_med = med_roll.update(roll)
                pitch_med = med_pitch.update(pitch)

                roll_kal = kal_roll.update(roll)
                pitch_kal = kal_pitch.update(pitch)

                jsonl_write(f, {
                    "t": float(t),
                    "ax": float(ax), "ay": float(ay), "az": float(az),
                    "roll_raw": float(roll), "pitch_raw": float(pitch),

                    "roll_sgd": roll_sgd,   "pitch_sgd": pitch_sgd,
                    "roll_ema": roll_ema,   "pitch_ema": pitch_ema,
                    "roll_med": roll_med,   "pitch_med": pitch_med,
                    "roll_kal": roll_kal,   "pitch_kal": pitch_kal,
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
