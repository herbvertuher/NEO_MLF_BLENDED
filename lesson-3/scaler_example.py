# %% 

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import matplotlib.pyplot as plt

# Генеруємо більший датасет з викидами
np.random.seed(42)
data = np.random.normal(loc=50, scale=10, size=100)  # Дані з нормальним розподілом
outliers = np.array([300, 350, 400])  # Викиди
data_with_outliers = np.concatenate([data, outliers])  # Додаємо викиди

# Перетворюємо в DataFrame
df = pd.DataFrame(data_with_outliers, columns=['feature'])

# Створюємо скейлери
scaler_standard = StandardScaler()
scaler_minmax = MinMaxScaler()
scaler_robust = RobustScaler()

# Масштабуємо дані
scaled_standard = scaler_standard.fit_transform(df)
scaled_minmax = scaler_minmax.fit_transform(df)
scaled_robust = scaler_robust.fit_transform(df)

# Додаємо результати в DataFrame для порівняння
df['StandardScaler'] = scaled_standard
df['MinMaxScaler'] = scaled_minmax
df['RobustScaler'] = scaled_robust

# Виведемо перші 5 значень
print(df.head(5))

# Візуалізація результатів
plt.figure(figsize=(10, 7))

plt.subplot(2, 2, 1)
plt.hist(df['feature'], bins=30, color='red', edgecolor='black')
plt.title('NonScaled')
plt.grid(True)

plt.subplot(2, 2, 2)
plt.hist(df['StandardScaler'], bins=30, color='skyblue', edgecolor='black')
plt.title('StandardScaler')
plt.grid(True)

plt.subplot(2, 2, 3)
plt.hist(df['MinMaxScaler'], bins=30, color='yellow', edgecolor='black')
plt.title('MinMaxScaler')
plt.grid(True)

plt.subplot(2, 2, 4)
plt.hist(df['RobustScaler'], bins=30, color='lightgreen', edgecolor='black')
plt.title('RobustScaler')
plt.grid(True)

plt.tight_layout()
plt.show()

# %%