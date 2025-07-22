# test.py

import numpy as np
import joblib
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
from sklearn.metrics import mean_squared_error

# --- Параметри ---
window_size = 11
signal_length = 3000
noise_std = 0.17

# --- Завантаження моделі та scaler ---
model = joblib.load('sgd_model.joblib')
scaler = joblib.load('scaler.joblib')
print("✅ Модель і scaler завантажено!")

# --- Генерація нового сигналу (наприклад, sin(x²)) ---
def add_impulse_noise(signal, impulse_ratio=0.01, impulse_strength=3.0):
    signal = signal.copy()
    n_impulses = int(len(signal) * impulse_ratio)
    indices = np.random.choice(len(signal), n_impulses, replace=False)
    
    # Генеруємо сплески: випадково -1 або +1
    impulses = np.random.choice([-1, 1], size=n_impulses) * impulse_strength
    signal[indices] += impulses
    return signal

x = np.linspace(0, 12, signal_length)
clean_signal = np.sin(x ** 2)
noisy_signal = clean_signal + np.random.normal(0, noise_std, size=signal_length)
noisy_signal = add_impulse_noise(noisy_signal)

# --- Обробка через модель ---
predicted_sgd = np.full(signal_length, np.nan)
last_pred = 0.0

for i in range(0, signal_length - window_size):
    window = noisy_signal[i:i + window_size]
    center = i + window_size // 2

    derivative = np.diff(window).mean()
    mean_val = np.mean(window)
    std_val = np.std(window)
    range_val = np.ptp(window)
    position = x[center] / x.max()

    if not np.isfinite(last_pred):
        last_pred = 0.0

    features = np.concatenate([
        window,
        [derivative, mean_val, std_val, range_val, last_pred, position]
    ])
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    try:
        features_scaled = scaler.transform([features])[0]
        y_pred = model.predict([features_scaled])[0]
    except Exception as e:
        print(f"⚠️ Пропущено i={i}: {e}")
        y_pred = 0.0

    last_pred = y_pred
    predicted_sgd[center] = y_pred



# --- Ковзне середнє ---
def ema_filter(signal, alpha=0.01):
    ema = np.zeros_like(signal)
    ema[0] = signal[0]
    for i in range(1, len(signal)):
        ema[i] = alpha * signal[i] + (1 - alpha) * ema[i - 1]
    return ema

filtered_avg = ema_filter(noisy_signal, 0.3)

# --- Медіанний фільтр ---
filtered_median = median_filter(noisy_signal, size=window_size)

# --- RMSE (лише по ненульових значеннях) ---
def compute_rmse(pred, target):
    mask = ~np.isnan(pred)
    return np.sqrt(mean_squared_error(target[mask], pred[mask]))

rmse_noise = compute_rmse(noisy_signal, clean_signal)
rmse_sgd = compute_rmse(predicted_sgd, clean_signal)
rmse_avg = compute_rmse(filtered_avg, clean_signal)
rmse_median = compute_rmse(filtered_median, clean_signal)

s = 0
for i in range(0, 10):
    s += noisy_signal[i]
s /= 11
print(s)

# --- Візуалізація ---
plt.figure(figsize=(16, 9))

# # 1. Шум
plt.subplot(4, 1, 1)
plt.title(f"Шумний сигнал — RMSE = {rmse_noise:.4f}")
plt.plot(clean_signal, label='Clean signal', linewidth=1)
plt.plot(noisy_signal, label='Noisy signal', alpha=0.5)
plt.legend()
plt.grid(True)

# 2. SGD
plt.subplot(4, 1, 2)
plt.title(f"SGDRegressor — RMSE = {rmse_sgd:.4f}")
plt.plot(clean_signal, label='Clean signal', linewidth=1)
plt.plot(noisy_signal, label='Noisy signal', alpha=0.3)
plt.plot(predicted_sgd, label='SGD output', linewidth=2)
plt.legend()
plt.grid(True)

# 3. Ковзне середнє
plt.subplot(4, 1, 3)
plt.title(f"EMA — RMSE = {rmse_avg:.4f}")
plt.plot(clean_signal, label='Clean signal', linewidth=1)
plt.plot(noisy_signal, label='Noisy signal', alpha=0.3)
plt.plot(filtered_avg, label='Averaged signal', linewidth=2)
plt.legend()
plt.grid(True)

# 4. Медіанний фільтр
plt.subplot(4, 1, 4)
plt.title(f"Медіанний фільтр — RMSE = {rmse_median:.4f}")
plt.plot(clean_signal, label='Clean signal', linewidth=1)
plt.plot(noisy_signal, label='Noisy signal', alpha=0.3)
plt.plot(filtered_median, label='Median filtered', linewidth=2)
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()