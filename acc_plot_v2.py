#!/usr/bin/env python3
# acc_plot_v2.py
# Побудова порівняльних графіків фільтрів за JSONL-логом.
# За один запуск створює ДВА окремих зображення: roll та pitch.

import os
import json
import numpy as np
import matplotlib.pyplot as plt


# =========================
# НАЛАШТУВАННЯ (редагуйте тут)
# =========================
LOG_PATH = "records/dynamic_test_2026-02-19_08-53-52.jsonl"  # <-- шлях до вашого jsonl
MAX_POINTS = 0              # 0 = без обмеження; інакше декімація до MAX_POINTS
SHOW_GRID = True
FIGSIZE = (14, 12)
SAVE_DPI = 150


# =========================
# Метрики
# =========================
def sd_from_mean(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    m = x.mean()
    return float(np.sqrt(np.mean((x - m) ** 2)))

def get_lag_samples(a: np.ndarray, b: np.ndarray) -> int:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a0 = a - a.mean()
    b0 = b - b.mean()
    corr = np.correlate(a0, b0, mode="full")
    return int(np.argmax(corr) - (len(a) - 1))


# =========================
# Читання JSONL
# =========================
def load_series_jsonl(path: str, axis: str):
    raw_key = f"{axis}_raw"
    sgd_key = f"{axis}_sgd"
    ema_key = f"{axis}_ema"
    med_key = f"{axis}_med"
    kal_key = f"{axis}_kal"

    raw, sgd, ema, med, kal = [], [], [], [], []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            raw.append(float(obj[raw_key]))
            sgd.append(float(obj[sgd_key]))
            ema.append(float(obj[ema_key]))
            med.append(float(obj[med_key]))
            kal.append(float(obj[kal_key]))

    return (
        np.asarray(raw, dtype=np.float64),
        np.asarray(sgd, dtype=np.float64),
        np.asarray(ema, dtype=np.float64),
        np.asarray(med, dtype=np.float64),
        np.asarray(kal, dtype=np.float64),
    )

def decimate_equal(*arrays, max_points: int):
    if max_points <= 0:
        return arrays
    n = len(arrays[0])
    if n <= max_points:
        return arrays
    idx = np.linspace(0, n - 1, max_points).astype(int)
    return tuple(a[idx] for a in arrays)


# =========================
# Плот
# =========================
def plot_filters(raw, sgd, ema, med, kal, title_prefix: str, save_path: str):
    r_sd = {
        "Raw": sd_from_mean(raw),
        "SGD": sd_from_mean(sgd),
        "EMA": sd_from_mean(ema),
        "Median": sd_from_mean(med),
        "Kalman": sd_from_mean(kal),
    }

    r_lag = {
        "SGD": get_lag_samples(sgd, raw),
        "EMA": get_lag_samples(ema, raw),
        "Median": get_lag_samples(med, raw),
        "Kalman": get_lag_samples(kal, raw),
    }

    fig, axs = plt.subplots(5, 1, figsize=FIGSIZE, sharex=True)

    axs[0].plot(raw, label=f"Raw ({title_prefix}) (SD={r_sd['Raw']:.2f}, N={len(raw)})", alpha=0.8)
    axs[0].set_title(f"{title_prefix}: Raw signal")
    axs[0].legend()
    if SHOW_GRID: axs[0].grid(True)

    axs[1].plot(raw, label="Raw", alpha=0.25)
    axs[1].plot(sgd, label=f"SGD (SD={r_sd['SGD']:.2f}, LAG={r_lag['SGD']} samples)", linewidth=1)
    axs[1].set_title("SGD filtered")
    axs[1].legend()
    if SHOW_GRID: axs[1].grid(True)

    axs[2].plot(raw, label="Raw", alpha=0.25)
    axs[2].plot(ema, label=f"EMA (SD={r_sd['EMA']:.2f}, LAG={r_lag['EMA']} samples)", linewidth=1)
    axs[2].set_title("EMA filtered")
    axs[2].legend()
    if SHOW_GRID: axs[2].grid(True)

    axs[3].plot(raw, label="Raw", alpha=0.25)
    axs[3].plot(med, label=f"Median (SD={r_sd['Median']:.2f}, LAG={r_lag['Median']} samples)", linewidth=1)
    axs[3].set_title("Median filtered")
    axs[3].legend()
    if SHOW_GRID: axs[3].grid(True)

    axs[4].plot(raw, label="Raw", alpha=0.25)
    axs[4].plot(kal, label=f"Kalman (SD={r_sd['Kalman']:.2f}, LAG={r_lag['Kalman']} samples)", linewidth=1)
    axs[4].set_title("Kalman filtered")
    axs[4].legend()
    if SHOW_GRID: axs[4].grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=SAVE_DPI)

    # maximize window if possible (optional)
    try:
        mng = plt.get_current_fig_manager()
        try:
            mng.window.state("zoomed")
        except Exception:
            try:
                mng.window.showMaximized()
            except Exception:
                pass
    except Exception:
        pass

    plt.show()


def main():
    if not os.path.exists(LOG_PATH):
        raise FileNotFoundError(f"File not found: {LOG_PATH}")

    base, _ = os.path.splitext(LOG_PATH)

    for axis in ("roll", "pitch"):
        raw, sgd, ema, med, kal = load_series_jsonl(LOG_PATH, axis)
        raw, sgd, ema, med, kal = decimate_equal(raw, sgd, ema, med, kal, max_points=MAX_POINTS)

        save_path = base + f"_{axis}_filters.png"
        plot_filters(raw, sgd, ema, med, kal, title_prefix=axis.upper(), save_path=save_path)


if __name__ == "__main__":
    main()
