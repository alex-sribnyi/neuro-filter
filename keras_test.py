# keras_test_optimized.py

import numpy as np
import matplotlib.pyplot as plt
import joblib
from keras.models import load_model
from sklearn.metrics import mean_squared_error
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints

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

# --- Імпульсний шум ---
# def add_impulse_noise(signal, ratio=0.01, strength=3.0):
#     signal = signal.copy()
#     idx = np.random.choice(len(signal), int(len(signal) * ratio), replace=False)
#     signal[idx] += np.random.choice([-1, 1], size=len(idx)) * strength
#     return signal

# --- RMSE ---
def compute_rmse(pred, target):
    mask = ~np.isnan(pred)
    return np.sqrt(mean_squared_error(target[mask], pred[mask]))

# --- Параметри ---
window_size = 25
signal_length = 3000
noise_std = 0.03

# --- Завантаження моделі та scaler ---
model = load_model("keras_model.keras")
scaler = joblib.load("keras_scaler.joblib")
target_scaler = joblib.load("keras_target_scaler.joblib")
print("Модель завантажено!")

# --- Генерація тестового сигналу ---
x = np.linspace(0, 12, signal_length)
clean_signal = np.sin(x) + 0.5 * np.sin(3 * x)
noisy_signal = clean_signal + np.random.normal(0, noise_std, size=signal_length)

# noisy_signal = add_impulse_noise(noisy_signal, ratio=0.01, strength=3.0)

# --- Побудова ознак для CNN (лише window) ---
X_test = []
centers = []
for i in range(0, signal_length - window_size):
    window = noisy_signal[i:i + window_size]
    features = np.nan_to_num(window, nan=0.0, posinf=0.0, neginf=0.0)
    X_test.append(features)
    centers.append(i + window_size // 2)

X_test = np.array(X_test)
X_test_scaled = scaler.transform(X_test)
X_test_cnn = X_test_scaled.reshape(-1, window_size, 1)

# --- Передбачення ---
y_pred_scaled = model.predict(X_test_cnn, verbose=0).ravel()
y_pred = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

predicted_keras = np.full(signal_length, np.nan)
predicted_keras[centers] = y_pred

# --- Інші фільтри ---
filtered_ema = ema_filter(noisy_signal, alpha=0.1)
filtered_median = median_filter(noisy_signal, window_size)
filtered_kalman = kalman_filter(noisy_signal, R=0.03, Q=0.05)

# --- RMSE ---
rmse_noise = compute_rmse(noisy_signal, clean_signal)
rmse_keras = compute_rmse(predicted_keras, clean_signal)
rmse_ema = compute_rmse(filtered_ema, clean_signal)
rmse_median = compute_rmse(filtered_median, clean_signal)
rmse_kalman = compute_rmse(filtered_kalman, clean_signal)

# --- Візуалізація ---
plt.figure(figsize=(16, 15))

plt.subplot(5, 1, 1)
plt.title(f"Шумний сигнал — RMSE = {rmse_noise:.4f}")
plt.plot(clean_signal, label='Clean', linewidth=1)
plt.plot(noisy_signal, label='Noisy', alpha=0.5)
plt.legend(); plt.grid()

plt.subplot(5, 1, 2)
plt.title(f"Keras MLP — RMSE = {rmse_keras:.4f}")
plt.plot(clean_signal, label='Clean', linewidth=1)
plt.plot(predicted_keras, label='Keras MLP', linewidth=2)
plt.legend(); plt.grid()

plt.subplot(5, 1, 3)
plt.title(f"EMA фільтр — RMSE = {rmse_ema:.4f}")
plt.plot(clean_signal, label='Clean', linewidth=1)
plt.plot(filtered_ema, label='EMA', linewidth=2)
plt.legend(); plt.grid()

plt.subplot(5, 1, 4)
plt.title(f"Медіанний фільтр — RMSE = {rmse_median:.4f}")
plt.plot(clean_signal, label='Clean', linewidth=1)
plt.plot(filtered_median, label='Median', linewidth=2)
plt.legend(); plt.grid()

plt.subplot(5, 1, 5)
plt.title(f"Фільтр Калмана — RMSE = {rmse_kalman:.4f}")
plt.plot(clean_signal, label='Clean', linewidth=1)
plt.plot(filtered_kalman, label='Kalman', linewidth=2)
plt.legend(); plt.grid()

plt.tight_layout()
plt.savefig("comparison_plot.png", dpi=150)
plt.show()
