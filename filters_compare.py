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

def rmse(true, pred):
    return np.sqrt(np.mean((true - pred) ** 2))

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
ema_pitch = ema_filter(pitch, alpha=0.1)

med_roll = median_filter(roll, window_size=11)
med_pitch = median_filter(pitch, window_size=11)

# === RMSE ===
r_rmse = {
    "SGD": rmse(roll, sgd_roll),
    "EMA": rmse(roll, ema_roll),
    "Median": rmse(roll, med_roll),
}
p_rmse = {
    "SGD": rmse(pitch, sgd_pitch),
    "EMA": rmse(pitch, ema_pitch),
    "Median": rmse(pitch, med_pitch),
}

# === Побудова графіків ===
fig, axs = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

# Roll
axs[0].plot(roll, label="Raw Roll", alpha=0.5)
axs[0].plot(sgd_roll, label=f"SGD Roll (RMSE={r_rmse['SGD']:.2f})", linewidth=1.5)
axs[0].plot(ema_roll, label=f"EMA Roll (RMSE={r_rmse['EMA']:.2f})", linestyle="--")
axs[0].plot(med_roll, label=f"Median Roll (RMSE={r_rmse['Median']:.2f})", linestyle="-.")
axs[0].set_title("Roll Signal Filtering")
axs[0].legend()
axs[0].grid(True)

# Pitch
axs[1].plot(pitch, label="Raw Pitch", alpha=0.5)
axs[1].plot(sgd_pitch, label=f"SGD Pitch (RMSE={p_rmse['SGD']:.2f})", linewidth=1.5)
axs[1].plot(ema_pitch, label=f"EMA Pitch (RMSE={p_rmse['EMA']:.2f})", linestyle="--")
axs[1].plot(med_pitch, label=f"Median Pitch (RMSE={p_rmse['Median']:.2f})", linestyle="-.")
axs[1].set_title("Pitch Signal Filtering")
axs[1].legend()
axs[1].grid(True)

plt.tight_layout()

# === Збереження ===
output_img = os.path.splitext(file_path)[0] + ".png"
plt.savefig(output_img)
plt.show()
