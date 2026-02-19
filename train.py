import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
import joblib

# --- Параметри ---
signal_length = 3000
window_size = 7
noise_std = 0.1
batch_size = 16

def add_impulse_noise(signal, impulse_ratio=0.01, impulse_strength=1.0):
    signal = signal.copy()
    n_impulses = int(len(signal) * impulse_ratio)
    indices = np.random.choice(len(signal), n_impulses, replace=False)
    
    # Генеруємо сплески: випадково -1 або +1
    impulses = np.random.choice([-1, 1], size=n_impulses) * impulse_strength
    signal[indices] += impulses
    return signal

# --- Генерація сигналу ---
x = np.linspace(0, 8 * np.pi, signal_length)
clean_signal = np.sin(x)
noisy_signal = add_impulse_noise(clean_signal + np.random.normal(0, noise_std, size=signal_length))

# --- Ініціалізація моделі та scaler'а ---
model = SGDRegressor(penalty='l2', alpha=1e-4, learning_rate='constant', eta0=1e-2,
                     max_iter=1, warm_start=True)
scaler = StandardScaler()

# --- Початкове навчання ---
X_init, y_init = [], []
for i in range(100):
    window = noisy_signal[i:i + window_size]
    derivative = np.diff(window).mean()
    autoreg = 0
    features = np.append(window, [derivative, autoreg])
    X_init.append(features)
    y_init.append(clean_signal[i + window_size // 2])

X_init = scaler.fit_transform(X_init)
model.partial_fit(X_init, y_init)
last_pred = 0

# --- Основний цикл онлайн-навчання ---
predicted = [None] * signal_length
buffer_X, buffer_y = [], []

for i in range(100, signal_length - window_size):
    window = noisy_signal[i:i + window_size]
    derivative = np.diff(window).mean()
    autoreg = last_pred

    features = np.append(window, [derivative, autoreg])
    features = scaler.transform([features])[0]

    y_true = clean_signal[i + window_size // 2]
    y_pred = model.predict([features])[0]
    last_pred = y_pred
    predicted[i + window_size // 2] = y_pred

    buffer_X.append(features)
    buffer_y.append(y_true)

    # Адаптація швидкості навчання
    if i == 500:
        model.eta0 = 5e-3
    elif i == 1000:
        model.eta0 = 1e-3

    # Часткове навчання на батчах
    if len(buffer_X) >= batch_size:
        model.partial_fit(np.array(buffer_X), np.array(buffer_y))
        buffer_X, buffer_y = [], []

# --- Збереження моделі ---
joblib.dump(model, "sgd_model.joblib")
joblib.dump(scaler, "scaler.joblib")
print("✅ Модель і scaler збережено!")

# --- Візуалізація результату навчання ---
plt.figure(figsize=(14, 5))
plt.plot(noisy_signal, label='Noisy', alpha=0.4)
plt.plot(clean_signal, label='Clean', linewidth=1)
plt.plot(predicted, label='Filtered (SGD)', linewidth=2)
plt.title('Онлайн-фільтрація сигналу (SGDRegressor)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
