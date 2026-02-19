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

# === RMSE ===
r_rmse = {
    "Raw": rmse(roll),
    "SGD": rmse(sgd_roll),
    "EMA": rmse(ema_roll),
    "Median": rmse(med_roll),
}

# === Variance ===
r_var = {
    "Raw": variance(roll),
    "SGD": variance(sgd_roll),
    "EMA": variance(ema_roll),
    "Median": variance(med_roll),
}

# === Вивід RMSE/Variance у термінал ===
print("Roll RMSE + Var:")
for k in r_rmse:
    print(f"{k}: RMSE = {r_rmse[k]:.4f}, Var = {r_var[k]:.4f}")

# === Побудова графіків ===
fig, axs = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

# --- Raw Roll ---
axs[0].plot(roll, label=f"Raw Roll (RMSE={r_rmse['Raw']:.2f}, Var={r_var['Raw']:.2f})", alpha=0.7)
axs[0].set_title("Raw Roll Signal")
axs[0].legend()
axs[0].grid(True)

# --- SGD ---
axs[1].plot(roll, label="Raw Roll", alpha=0.3)
axs[1].plot(sgd_roll, label=f"SGD Roll (RMSE={r_rmse['SGD']:.2f}, Var={r_var['SGD']:.2f})", linewidth=1.5)
axs[1].set_title("SGD Filtered Roll")
axs[1].legend()
axs[1].grid(True)

# --- EMA ---
axs[2].plot(roll, label="Raw Roll", alpha=0.3)
axs[2].plot(ema_roll, label=f"EMA Roll (RMSE={r_rmse['EMA']:.2f}, Var={r_var['EMA']:.2f})", linestyle="--")
axs[2].set_title("EMA Filtered Roll")
axs[2].legend()
axs[2].grid(True)

# --- Median ---
axs[3].plot(roll, label="Raw Roll", alpha=0.3)
axs[3].plot(med_roll, label=f"Median Roll (RMSE={r_rmse['Median']:.2f}, Var={r_var['Median']:.2f})", linestyle="-.")
axs[3].set_title("Median Filtered Roll")
axs[3].legend()
axs[3].grid(True)

plt.tight_layout()
plt.savefig(os.path.splitext(file_path)[0] + "_filters.png")
plt.show()

