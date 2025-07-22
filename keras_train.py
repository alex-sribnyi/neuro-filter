# keras_train_optimized.py

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, InputLayer
from keras.regularizers import l2
from keras.losses import Huber
from keras.optimizers import Nadam
from sklearn.preprocessing import StandardScaler
import joblib

# --- Параметри ---
signal_length = 5000
window_size = 25
noise_std = 0.3

# --- Генерація сигналу ---
x = np.linspace(0, 10 * np.pi, signal_length)
clean_signal = np.sin(x)
noisy_signal = clean_signal + np.random.normal(0, noise_std, size=signal_length)

# --- Побудова ознак ---
X, y = [], []
for i in range(signal_length - window_size):
    window = noisy_signal[i:i + window_size]
    center = i + window_size // 2

    derivative = np.diff(window).mean()
    mean_val = np.mean(window)
    std_val = np.std(window)
    range_val = np.ptp(window)
    position = x[center] / x.max()
    autoreg = 0  # при тренуванні не використовуємо autoreg

    features = np.concatenate([window, [derivative, mean_val, std_val, range_val, autoreg, position]])
    X.append(features)
    y.append(clean_signal[center])

X = np.array(X)
y = np.array(y)

# --- Масштабування ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, "keras_scaler.joblib")

# --- Масштабування цілі ---
target_scaler = StandardScaler()
y_scaled = target_scaler.fit_transform(y.reshape(-1, 1)).ravel()
joblib.dump(target_scaler, "keras_target_scaler.joblib")

# --- Побудова моделі ---
model = Sequential()
model.add(InputLayer(input_shape=(X.shape[1],)))
model.add(Dense(128, activation='relu', kernel_regularizer=l2(1e-5)))
model.add(Dropout(0.05))
model.add(Dense(64, activation='relu', kernel_regularizer=l2(1e-5)))
model.add(Dropout(0.05))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

model.compile(optimizer=Nadam(), loss=Huber(delta=1.0))

# --- Навчання ---
model.fit(X_scaled, y_scaled, epochs=100, batch_size=32, verbose=1)

# --- Збереження ---
model.save("keras_model.keras")
print("✅ Оптимізована модель збережена як keras_model.keras")
