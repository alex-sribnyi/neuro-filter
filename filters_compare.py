import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# === Фільтри ===
def ema_filter(signal, alpha=0.1):
    ema = np.zeros_like(signal)
    ema[0] = signal[0]
    for i in range(1, len(signal)):
        ema[i] = alpha * signal[i] + (1 - alpha) * ema[i - 1]
    return ema

def median_filter(signal, window_size):
    if window_size % 2 == 0:
        window_size += 1
    filtered = np.zeros_like(signal)
    margin = window_size // 2
    for i in range(margin, len(signal) - margin):
        window = np.sort(signal[i - margin : i + margin + 1])
        filtered[i] = window[margin]
    filtered[:margin] = filtered[margin]
    filtered[-margin:] = filtered[-margin - 1]
    return filtered

def rmse(signal):
    mean = np.mean(signal)
    return np.sqrt(np.mean((signal - mean) ** 2))

def kalman_filter(
    measurements,
    process_var=1e-5,       # дисперсія процесу Q (наскільки швидко може змінюватися істинний сигнал)
    meas_var=1e-2,          # дисперсія вимірювань R (рівень шуму датчика)
    x0=None,                # початкова оцінка стану
    P0=1.0                  # початкова дисперсія помилки оцінки
):
    """
    Фільтрація зашумленого 1D сигналу методом Калмана
    з моделлю x_k = x_{k-1}, z_k = x_k + noise.

    :param measurements: iterable / список виміряних значень (шумний сигнал)
    :param process_var: дисперсія процесу Q
    :param meas_var: дисперсія вимірювань R
    :param x0: початкова оцінка стану (якщо None — береться перше вимірювання)
    :param P0: початкова дисперсія помилки оцінки
    :return: список відфільтрованих значень тієї ж довжини
    """
    measurements = list(measurements)
    if not measurements:
        return []

    # Ініціалізація
    x = measurements[0] if x0 is None else x0  # початкова оцінка стану
    P = P0                                     # початкова помилка оцінки

    Q = process_var    # дисперсія процесу
    R = meas_var       # дисперсія вимірювань

    filtered = []

    for z in measurements:
        # ----- КРОК ПРОГНОЗУ -----
        # Модель: x_k = x_{k-1}, отже прогноз той самий
        x_pred = x
        P_pred = P + Q   # до помилки додаємо невизначеність процесу

        # ----- КРОК ОНОВЛЕННЯ -----
        # Калманівський коефіцієнт
        K = P_pred / (P_pred + R)

        # Оновлена оцінка стану
        x = x_pred + K * (z - x_pred)

        # Оновлена помилка оцінки
        P = (1 - K) * P_pred

        filtered.append(x)

    return np.array(filtered)

def variance(signal):
    return float(np.var(signal))

# === Аргумент командного рядка ===
if len(sys.argv) < 2:
    print("Вкажіть шлях до .txt файлу як аргумент запуску.")
    sys.exit(1)

file_path = sys.argv[1]

if not os.path.exists(file_path):
    print(f"Файл не знайдено: {file_path}")
    sys.exit(1)

# === Завантаження даних ===
data = np.loadtxt(file_path, delimiter=",")
roll, pitch = data[:, 0], data[:, 1]
sgd_roll, sgd_pitch = data[:, 2], data[:, 3]

# === Фільтрація ===
ema_roll = ema_filter(roll, alpha=0.1)
med_roll = median_filter(roll, window_size=11)
kal_roll = kalman_filter(roll, process_var=1e-5, meas_var=3e-4)

def get_lag(signal_a, signal_b):
    corr = np.correlate(signal_a - signal_a.mean(), signal_b - signal_b.mean(), mode='full')
    return np.argmax(corr) - (len(signal_a) - 1)    
    
# === RMSE ===
r_rmse = {
    "Raw": rmse(roll),
    "SGD": rmse(sgd_roll),
    "EMA": rmse(ema_roll),
    "Median": rmse(med_roll),
    "Kalman": rmse(kal_roll),
}

# === RMSE ===
r_lag = {
    "SGD": get_lag(sgd_roll, roll),
    "EMA": get_lag(ema_roll, roll),
    "Median": get_lag(med_roll, roll),
    "Kalman": get_lag(kal_roll, roll),
}

# === Вивід RMSE/Variance у термінал ===
print("Roll RMSE + Var:")
for k in r_rmse:
    print(f"{k}: RMSE = {r_rmse[k]:.4f}")

# === Побудова графіків ===
fig, axs = plt.subplots(5, 1, figsize=(14, 12), sharex=True)

# --- Raw Roll ---
axs[0].plot(roll, label=f"Raw Roll (RMSE={r_rmse['Raw']:.2f}, Length={len(roll)}", alpha=0.7)
axs[0].set_title("Raw Roll Signal")
axs[0].legend()
axs[0].grid(True)

# --- SGD ---
axs[1].plot(roll, label="Raw Roll", alpha=0.3)
axs[1].plot(sgd_roll, label=f"SGD Roll (RMSE={r_rmse['SGD']:.2f}, LAG={r_lag['SGD']:.2f})", linewidth=1)
axs[1].set_title("SGD Filtered Roll")
axs[1].legend()
axs[1].grid(True)

# --- EMA ---
axs[2].plot(roll, label="Raw Roll", alpha=0.3)
axs[2].plot(ema_roll, label=f"EMA Roll (RMSE={r_rmse['EMA']:.2f}, LAG={r_lag['EMA']:.2f})", linewidth=1)
axs[2].set_title("EMA Filtered Roll")
axs[2].legend()
axs[2].grid(True)

# --- Median ---
axs[3].plot(roll, label="Raw Roll", alpha=0.3)
axs[3].plot(med_roll, label=f"Median Roll (RMSE={r_rmse['Median']:.2f}, LAG={r_lag['Median']:.2f})", linewidth=1)
axs[3].set_title("Median Filtered Roll")
axs[3].legend()
axs[3].grid(True)

# --- Kalman ---
axs[4].plot(roll, label="Raw Roll", alpha=0.3)
axs[4].plot(kal_roll, label=f"Kalman Roll (RMSE={r_rmse['Kalman']:.2f}, LAG={r_lag['Kalman']:.2f})", linewidth=1)
axs[4].set_title("Kalman Filtered Roll")
axs[4].legend()
axs[4].grid(True)

mng = plt.get_current_fig_manager()
try:
    # TkAgg (часто в Windows / IDLE / Spyder)
    mng.window.state('zoomed')
except Exception:
    try:
        # Qt5Agg / QtAgg
        mng.window.showMaximized()
    except Exception:
        try:
            # wxAgg
            mng.frame.Maximize(True)
        except Exception:
            pass  # якщо не вдалось — просто ігноруємо

plt.tight_layout()
plt.savefig(os.path.splitext(file_path)[0] + "_filters.png")
plt.show()

