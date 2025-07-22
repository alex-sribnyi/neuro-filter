# train.py (покращена версія)

import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler

# --- Параметри ---
window_size = 11
signal_length = 3000
noise_std = 0.17

# --- Генерація сигналу ---
def add_impulse_noise(signal, impulse_ratio=0.01, impulse_strength=3.0):
    signal = signal.copy()
    n_impulses = int(len(signal) * impulse_ratio)
    indices = np.random.choice(len(signal), n_impulses, replace=False)
    
    # Генеруємо сплески: випадково -1 або +1
    impulses = np.random.choice([-1, 1], size=n_impulses) * impulse_strength
    signal[indices] += impulses
    return signal

x = np.linspace(0, 8 * np.pi, signal_length)
clean_signal = np.sin(x)
noisy_signal = clean_signal + np.random.normal(0, noise_std, size=signal_length)
noisy_signal = add_impulse_noise(noisy_signal)

# --- Ініціалізація моделі ---
model = SGDRegressor(
    penalty='l2',
    alpha=1e-4,
    learning_rate='constant',
    eta0=1e-2,
    max_iter=1000,
    tol=1e-4
)

scaler = StandardScaler()
predicted = [None] * signal_length
features_list, targets = [], []

# --- Формування повної вибірки для навчання ---
for i in range(0, signal_length - window_size):
    window = noisy_signal[i:i + window_size]
    center = i + window_size // 2

    # Ознаки:
    derivative = np.diff(window).mean()
    mean_val = np.mean(window)
    std_val = np.std(window)
    range_val = np.ptp(window)
    autoreg = 0 if i == 0 else predicted[center - 1] or 0
    position = x[center] / x.max()  # нормалізована позиція

    # Формування вектора ознак
    features = np.concatenate([
        window,
        [derivative, mean_val, std_val, range_val, autoreg, position]
    ])

    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    features_list.append(features)
    targets.append(clean_signal[center])

# --- Перетворення у масиви ---
X = np.array(features_list)
y = np.array(targets)

# --- Масштабування ознак ---
X = scaler.fit_transform(X)

# --- Навчання моделі ---
model.fit(X, y)

# --- Прогнозування ---
predicted_values = model.predict(X)
for i, val in enumerate(predicted_values):
    predicted[i + window_size // 2] = val

# --- Збереження ---
joblib.dump(model, 'sgd_model.joblib')
joblib.dump(scaler, 'scaler.joblib')
print("✅ Модель і scaler збережено!")

# --- Візуалізація ---
plt.figure(figsize=(14, 5))
plt.plot(noisy_signal, label='Noisy', alpha=0.4)
plt.plot(clean_signal, label='Clean', linewidth=1)
plt.plot(predicted, label='Filtered (SGD Improved)', linewidth=2)
plt.title('Покращене навчання на синусоїді (SGDRegressor)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
