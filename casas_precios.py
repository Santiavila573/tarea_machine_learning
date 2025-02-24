# Paso 1: Importar librerías necesarias
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Configurar matplotlib para mostrar gráficos en entornos no interactivos
plt.ion()  # Activa el modo interactivo
plt.switch_backend('TkAgg')  # Usa TkAgg como backend (puedes cambiar a 'Qt5Agg' si prefieres)

# Paso 2: Cargar el dataset (usamos California Housing en lugar de Boston)
california = fetch_california_housing()
data = pd.DataFrame(california.data, columns=california.feature_names)
data['PRICE'] = california.target

# Paso 3: Análisis Exploratorio de Datos (EDA)
print("Primeras 5 filas del dataset:")
print(data.head())
print("\nInformación del dataset:")
print(data.info())
print("\nEstadísticas descriptivas:")
print(data.describe())

# Visualización de la correlación entre variables
plt.figure(figsize=(12, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title('Matriz de Correlación')
plt.show()  # Muestra el gráfico inmediatamente

# Distribución del precio (variable objetivo)
plt.figure(figsize=(10, 6))
sns.histplot(data['PRICE'], bins=30)
plt.title('Distribución de Precios de Viviendas')
plt.xlabel('Precio')
plt.show()  # Muestra el gráfico inmediatamente

# Paso 4: Preparar los datos
X = data.drop('PRICE', axis=1)  # Variables independientes
y = data['PRICE']  # Variable dependiente

# Dividir en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Imprimir una muestra de los datos divididos
print("\nMuestra de X_train (primeras 5 filas):")
print(X_train.head())
print("\nMuestra de X_test (primeras 5 filas):")
print(X_test.head())

# Paso 5: Crear y entrenar el modelo
model = LinearRegression()
model.fit(X_train, y_train)

# Paso 6: Hacer predicciones
y_pred = model.predict(X_test)

# Paso 7: Evaluar el modelo
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nResultados de la evaluación:")
print(f"Error Cuadrático Medio (MSE): {mse:.2f}")
print(f"Raíz del Error Cuadrático Medio (RMSE): {rmse:.2f}")
print(f"Coeficiente de Determinación (R²): {r2:.2f}")

# Visualización de predicciones vs valores reales
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Valores Reales')
plt.ylabel('Predicciones')
plt.title('Predicciones vs Valores Reales')
plt.show()  # Muestra el gráfico inmediatamente

# Paso 8: Mostrar los coeficientes del modelo
coeficientes = pd.DataFrame(model.coef_, X.columns, columns=['Coeficiente'])
print("\nCoeficientes del modelo:")
print(coeficientes)