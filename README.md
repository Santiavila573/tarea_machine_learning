# Predicción de Precios de Viviendas en California

Este proyecto utiliza un modelo de regresión lineal para predecir los precios de viviendas en California a partir del conjunto de datos **California Housing**, disponible en `scikit-learn`. Incluye análisis exploratorio de datos (EDA), entrenamiento del modelo, evaluación y visualización de resultados.

## Tabla de Contenidos
- [Descripción](#descripción)
- [Requisitos](#requisitos)
- [Instalación](#instalación)
- [Uso](#uso)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Resultados](#resultados)
- [Limitaciones](#limitaciones)
- [Contribuciones](#contribuciones)
- [Licencia](#licencia)

## Descripción
El objetivo es predecir los precios de viviendas (`PRICE`) basándose en características como el ingreso medio (`MedInc`), la edad de las casas (`HouseAge`), el número promedio de habitaciones (`AveRooms`), entre otras. El programa realiza:
1. Carga y exploración del dataset.
2. Preparación de datos (división en entrenamiento y prueba).
3. Entrenamiento de un modelo de regresión lineal.
4. Evaluación con métricas como MSE, RMSE y R².
5. Visualización de correlaciones, distribuciones y predicciones.

## Requisitos
- **Python**: 3.7 o superior
- **Librerías**:
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `seaborn`
  - `scikit-learn`

## python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

## Instala las dependencias:

pip install -r requirements.txt

## Si no hay un requirements.txt, instala manualmente:

pip install numpy pandas matplotlib seaborn scikit-learn

## Uso

Asegúrate de estar en el directorio del proyecto y tener el entorno activado.
Ejecuta el script principal

python california_housing_regression.py
