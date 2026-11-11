"""Generación reproducible de dos clases linealmente separables en R²."""
import numpy as np

def generar_datos():
    """Crea dos clases separables en R² con semilla reproducible."""
    rng = np.random.default_rng(12)
    negativos = rng.normal(loc=(-2.0, -1.5), scale=0.35, size=(12, 2))
    positivos = rng.normal(loc=(2.0, 1.5), scale=0.35, size=(12, 2))
    X = np.vstack((negativos, positivos))
    y = np.r_[-np.ones(len(negativos)), np.ones(len(positivos))]
    return X, y
