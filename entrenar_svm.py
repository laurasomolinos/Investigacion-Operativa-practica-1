"""Entrena una SVM gaussiana (RBF) sobre MNIST, en CPU y con semilla fija."""

from pathlib import Path
import random
import time

import joblib 
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from torchvision import datasets, transforms

SEMILLA = 42
MUESTRAS_ENTRENAMIENTO = 12000  #para no disparar el coste de SVC en CPU se divide el conjunto de entrenamiento 
C = 10.0 #regulación de penalización de violaciones de margen 
GAMMA = "scale" #control del nucleo gaussiano: "scale" = 1/(n_features * X.var()) = 1/(784 * 0.083) = 0.0154
CARPETA = Path(__file__).resolve().parent


def cargar_datos():
    """Carga las mismas imágenes de MNIST que usa nuestra CNN."""
    transformar = transforms.ToTensor() #imagenes a tensores y escala pixeles a [0, 1]
    entrenamiento = datasets.MNIST(CARPETA / "datos", train=True, download=True, transform=transformar) #60000 imágenes -> train 
    prueba = datasets.MNIST(CARPETA / "datos", train=False, download=True, transform=transformar) #10000 imágenes -> test

    # SVC necesita una fila por imagen y una columna por píxel (784 características).
    # ToTensor convierte los píxeles de 0..255 al rango 0..1.
    rng = np.random.default_rng(SEMILLA) 
    indices = rng.choice(len(entrenamiento), MUESTRAS_ENTRENAMIENTO, replace=False) #elegir las 12000 imagenes sin reemplazo
    #dividimos conjunto de entrenamiento y test: 
    # - cada imagen de 28x28 a vector de 784 elementos (12000,28,28) -> (12000,784)
    X_train = entrenamiento.data[indices].numpy().reshape(-1, 784).astype(np.float32) / 255.0
    #etiquetas reales de las 12000 imagenes elegidas
    y_train = entrenamiento.targets[indices].numpy()
    # -misma transformación para test (10000,28,28) -> (10000,784)
    X_test = prueba.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
    y_test = prueba.targets.numpy()
    return X_train, y_train, X_test, y_test


def main():
    random.seed(SEMILLA)
    np.random.seed(SEMILLA)

    #cargar datos train y test 
    X_train, y_train, X_test, y_test = cargar_datos()
    print(f"Entrenamiento SVM: {X_train.shape[0]} imágenes y {X_train.shape[1]} píxeles por imagen")
    print(f"Test: {X_test.shape[0]} imágenes")
    print(f"Kernel RBF | C={C} | gamma={GAMMA}")

    #Configuramos y creamos la SVM con kernel gaussiano 
    #kernel ="rbf" SVM construya fronteras no lineales
    modelo = SVC(kernel="rbf", C=C, gamma=GAMMA, cache_size=512)
    inicio = time.perf_counter() #contamos lo que tarda en entrenar
    modelo.fit(X_train, y_train)
    print(f"Entrenamiento finalizado. Tiempo: {time.perf_counter()-inicio:.1f} s")

    # El conjunto de test se usa únicamente para evaluar el modelo ya entrenado
    predicciones = modelo.predict(X_test)
    precision = 100 * accuracy_score(y_test, predicciones)
    print(f"Acierto test SVM: {precision:.2f}%")
    print(f"Vectores de soporte: {len(modelo.support_)}") #observaciones que intervienen en la funcion de decisión

    ruta = CARPETA / "svm_rbf_mnist.joblib"
    joblib.dump(modelo, ruta, compress=3)
    print(f"Modelo guardado en: {ruta}") #guardamos el modelo comprimido 


if __name__ == "__main__":
    main()
