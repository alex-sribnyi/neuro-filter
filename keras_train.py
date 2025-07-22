# keras_train_cnn.py (навчання лише на синусі з імпульсним шумом)

import numpy as np
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
import joblib

# --- Параметри ---
window_size = 25
signal_length = 3000
noise_std = 0.3

# --- Генерація синусоїдального сигналу ---
def generate_signal(length):
    x = np.linspace(0, 12, length)
    return np.sin(x)

X = []
y = []

for _ in range(600):
    signal = generate_signal(window_size + 1)
    noise = np.random.normal(0, noise_std, size=signal.shape)

    noisy = signal + noise

    for i in range(len(noisy) - window_size):
        window = noisy[i:i + window_size]
        target = signal[i + window_size // 2]  # середина як таргет
        X.append(window)
        y.append(target)

X = np.array(X)
y = np.array(y).reshape(-1, 1)

# --- Масштабування ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, "keras_scaler.joblib")

scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y)
joblib.dump(scaler_y, "keras_target_scaler.joblib")

# --- CNN модель ---
model = keras.Sequential([
    keras.layers.Input(shape=(window_size, 1)),
    keras.layers.Conv1D(32, kernel_size=3, activation='relu'),
    keras.layers.Conv1D(64, kernel_size=3, activation='relu'),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse')
model.summary()

# --- Навчання ---
X_cnn = X_scaled.reshape(-1, window_size, 1)
model.fit(X_cnn, y_scaled, epochs=100, batch_size=32, verbose=1)

# --- Збереження ---
model.save("keras_model.keras")
print("✅ CNN модель та scaler'и збережено!")