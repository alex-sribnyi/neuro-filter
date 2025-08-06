# test.py

import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints

# --- Параметри ---
window_size = 7
signal_length = 3000
noise_std = 0.01

# --- Завантаження моделі та scaler ---
model = joblib.load('sgd_model.joblib')
scaler = joblib.load('scaler.joblib')
print("✅ Модель і scaler завантажено!")

# --- EMA фільтр ---
def ema_filter(signal, alpha=0.1):
    ema = np.zeros_like(signal)
    ema[0] = signal[0]
    for i in range(1, len(signal)):
        ema[i] = alpha * signal[i] + (1 - alpha) * ema[i - 1]
    return ema

# --- Медіанний фільтр ---
def median_filter(signal, window_size):
    if window_size % 2 == 0: window_size += 1
    filtered = np.zeros_like(signal)
    margin = window_size // 2
    for i in range(margin, len(signal) - margin):
        window = np.array(signal[i - margin : i + margin + 1])
        filtered[i] = np.sort(window)[margin]
        if i == margin:
            filtered[:margin] = filtered[margin]
        elif i == len(signal) - margin - 1:
            filtered[-margin:] = filtered[-margin - 1]
    return filtered

# --- Калманівський фільтр ---
def fx(x, dt):
    return np.array([
        x[0] + dt * x[1] + 0.5 * dt**2 * x[2],
        x[1] + dt * x[2],
        x[2]
    ])

def hx(x):
    return np.array([x[0]])

def kalman_filter(signal, dt=1.0, R=0.05, Q=1e-2):
    sigmas = MerweScaledSigmaPoints(n=3, alpha=0.3, beta=2.0, kappa=0.0)
    ukf = UKF(dim_x=3, dim_z=1, fx=fx, hx=hx, dt=dt, points=sigmas)
    ukf.x = np.array([0., 0., 0.])
    ukf.P *= 10.
    ukf.R *= R
    ukf.Q *= Q * np.eye(3)

    filtered = []
    for z in signal:
        ukf.predict()
        ukf.update(np.array([z]))
        filtered.append(ukf.x[0])
    return np.array(filtered)

# --- Генерація нового сигналу (наприклад, sin(x²)) ---
# def add_impulse_noise(signal, impulse_ratio=0.01, impulse_strength=3.0):
#     signal = signal.copy()
#     n_impulses = int(len(signal) * impulse_ratio)
#     indices = np.random.choice(len(signal), n_impulses, replace=False)
    
#     # Генеруємо сплески: випадково -1 або +1
#     impulses = np.random.choice([-1, 1], size=n_impulses) * impulse_strength
#     signal[indices] += impulses
#     return signal

x = np.linspace(0, 12, signal_length)
clean_signal = np.sin(x) + 0.5 * np.sin(3 * x)
# clean_signal = np.zeros(signal_length)  # Зворотний лінійний спад
noisy_signal = clean_signal + np.random.normal(0, noise_std, size=signal_length)
# noisy_signal = add_impulse_noise(noisy_signal)

# --- Обробка через модель ---
predicted_sgd = np.full(signal_length, np.nan)
last_pred = 0.0

for i in range(0, signal_length - window_size):
    window = noisy_signal[i:i + window_size]
    center = i + window_size // 2

    derivative = np.diff(window).mean()
    autoreg = last_pred

    features = np.append(window, [derivative, autoreg])
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    try:
        features_scaled = scaler.transform([features])[0]
        y_pred = model.predict([features_scaled])[0]
    except Exception as e:
        print(f"⚠️ Пропущено i={i}: {e}")
        y_pred = 0.0

    last_pred = y_pred
    predicted_sgd[center] = y_pred

# --- EMA фільтр ---
filtered_ema = ema_filter(noisy_signal, 0.05)

# --- Медіанний фільтр ---
filtered_median = median_filter(noisy_signal, window_size)

# --- Калманівський фільтр ---
filtered_kalman = kalman_filter(noisy_signal, R=0.03, Q=0.05)   

# --- RMSE (лише по ненульових значеннях) ---
def compute_rmse(pred, target):
    mask = ~np.isnan(pred)
    return np.sqrt(mean_squared_error(target[mask], pred[mask]))

rmse_noise = compute_rmse(noisy_signal, clean_signal)
rmse_sgd = compute_rmse(predicted_sgd, clean_signal)
rmse_avg = compute_rmse(filtered_ema, clean_signal)
rmse_median = compute_rmse(filtered_median, clean_signal)
rmse_kalman = compute_rmse(filtered_kalman, clean_signal)

s = 0
for i in range(0, 10):
    s += noisy_signal[i]
s /= 11
print(s)

# --- Візуалізація ---
plt.figure(figsize=(16, 15))

# # 1. Шум
plt.subplot(5, 1, 1)
plt.title(f"Шумний сигнал — RMSE = {rmse_noise:.4f}")
plt.plot(clean_signal, label='Clean signal', linewidth=1)
plt.plot(noisy_signal, label='Noisy signal', alpha=0.5)
plt.legend()
plt.grid(True)

# 2. SGD
plt.subplot(5, 1, 2)
plt.title(f"SGDRegressor — RMSE = {rmse_sgd:.4f}")
plt.plot(clean_signal, label='Clean signal', linewidth=1)
plt.plot(noisy_signal, label='Noisy signal', alpha=0.3)
plt.plot(predicted_sgd, label='SGD output', linewidth=2)
plt.legend()
plt.grid(True)

# 3. EMA фільтр
plt.subplot(5, 1, 3)
plt.title(f"EMA — RMSE = {rmse_avg:.4f}")
plt.plot(clean_signal, label='Clean signal', linewidth=1)
plt.plot(noisy_signal, label='Noisy signal', alpha=0.3)
plt.plot(filtered_ema, label='Averaged signal', linewidth=2)
plt.legend()
plt.grid(True)

# 4. Медіанний фільтр
plt.subplot(5, 1, 4)
plt.title(f"Медіанний фільтр — RMSE = {rmse_median:.4f}")
plt.plot(clean_signal, label='Clean signal', linewidth=1)
plt.plot(noisy_signal, label='Noisy signal', alpha=0.3)
plt.plot(filtered_median, label='Median filtered', linewidth=2)
plt.legend()
plt.grid(True)

# 5. Калманівський фільтр
plt.subplot(5, 1, 5)
plt.title(f"Калманівський фільтр — RMSE = {rmse_kalman:.4f}")
plt.plot(clean_signal, label='Clean signal', linewidth=1)
plt.plot(noisy_signal, label='Noisy signal', alpha=0.3)
plt.plot(filtered_kalman, label='Kalman filtered', linewidth=2)
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()