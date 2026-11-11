"""Apartado 1: entrena una CNN para reconocer los dígitos del conjunto MNIST.

Ejecutar:  python entrenar_cnn.py     (guarda cnn_mnist.pth)

Se utiliza PyTorch, con semillas fijas para facilitar la reproducción del experimento.
El entrenamiento se realiza en CPU y guarda los pesos para usarlos después en FGSM.

Arquitectura (105.866 parámetros):
  Conv(1→16, 3x3) → ReLU → MaxPool  →  Conv(16→32, 3x3) → ReLU → MaxPool  →
  Flatten(1568) → Linear(1568→64) → ReLU → Linear(64→10)
  Parámetros: 160 + 4.640 + 100.416 + 650 = 105.866.
Entrenamiento: Adam (lr 1e-3), lote 128, 4 épocas, semilla 42.

Nota de reproducibilidad: con la misma semilla el resultado es el mismo en una misma máquina
y versión de PyTorch, pero puede diferir unas décimas entre equipos o versiones (por eso se
entregan también los pesos ya entrenados, cnn_mnist.pth).
"""

import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

#configuración del experimento.
SEMILLA = 42
EPOCAS = 4
LOTE = 128  #(batch size):num imagenes que procesa la RN antes de actualizar sus pesos 
TASA_APRENDIZAJE = 0.001
CARPETA = Path(__file__).resolve().parent  #carpeta que contiene este archivo.


def fijar_semillas(semilla: int) -> None:
    """Fija las fuentes de aleatoriedad para favorecer resultados reproducibles."""
    random.seed(semilla)   #para python
    np.random.seed(semilla)  #para numpy
    torch.manual_seed(semilla) #para PyTorch
    torch.use_deterministic_algorithms(True) #obliga a PyTorch a usar solo operaciones deterministas (lanza error si alguna no lo es)


class CNNMNIST(nn.Module):
    """CNN con dos convoluciones y dos capas lineales para clasificar 10 dígitos.
    Las dimensiones espaciales evolucionan de 28x28 a 14x14 y luego a 7x7.
    La última capa produce una puntuación (logit) para cada clase, de 0 a 9.
    """

    def __init__(self) -> None:
        super().__init__()

        # Extracción de características:
        # [N, 1, 28, 28] -> [N, 16, 28, 28] -> [N, 16, 14, 14]
        #               -> [N, 32, 14, 14] -> [N, 32, 7, 7].
        # Conv2d aprende filtros; ReLU introduce no linealidad;
        # MaxPool2d reduce a la mitad el alto y el ancho.
        self.caracteristicas = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        # Clasificador:
        # Flatten transforma los 32 mapas de 7x7 en 1.568 valores.
        # La capa oculta tiene 64 neuronas y la salida tiene 10 logits.
        self.clasificador = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, imagenes: torch.Tensor) -> torch.Tensor:
        """Realiza la propagación hacia delante y devuelve los diez logits."""
        # No aplicamos softmax: CrossEntropyLoss admite logits directamente.
        return self.clasificador(self.caracteristicas(imagenes)) 


def evaluar(modelo: nn.Module, datos: DataLoader) -> float:
    """Calcula la precisión, en porcentaje, sobre las imágenes recibidas."""
    modelo.eval()  # Activa el modo evaluación (útil si hubiera dropout o batch norm).
    aciertos = 0
    total = 0

    # En evaluación no hace falta almacenar gradientes: ahorra memoria y cálculo.
    with torch.no_grad():
        for imagenes, etiquetas in datos:
            # argmax elige el índice de la puntuación más alta entre las 10 clases.
            predicciones = modelo(imagenes).argmax(dim=1)
            aciertos += (predicciones == etiquetas).sum().item()
            total += etiquetas.size(0)

    return 100 * aciertos / total


def main() -> None:
    """Prepara los datos, entrena la CNN, evalúa su precisión y guarda los pesos."""
    fijar_semillas(SEMILLA)
    CARPETA.mkdir(parents=True, exist_ok=True)

    # ToTensor convierte cada imagen MNIST en un tensor [1, 28, 28],
    # escalando sus píxeles de 0 a 255 al intervalo [0, 1].
    transformar = transforms.ToTensor()

    # MNIST dispone de 60.000 imágenes de entrenamiento y 10.000 de prueba.
    entrenamiento = datasets.MNIST(
        root=CARPETA / "datos", train=True, download=True, transform=transformar
    )
    prueba = datasets.MNIST(
        root=CARPETA / "datos", train=False, download=True, transform=transformar
    )

    # Distinto orden (shuffle=True) 
    #El generador con semilla fija permite reproducir el orden de los lotes.
    generador = torch.Generator().manual_seed(SEMILLA)
    cargador_entrenamiento = DataLoader(
        entrenamiento, batch_size=LOTE, shuffle=True, num_workers=0,
        generator=generador,
    )
    cargador_prueba = DataLoader(
        prueba, batch_size=256, shuffle=False, num_workers=0
    )

    modelo = CNNMNIST()  # En este script se utiliza CPU.
    #Cross entropy: mide el error entre las puntuaciones y las clases reales.
    perdida = nn.CrossEntropyLoss()
    # Adam adapta el tamaño de las actualizaciones a partir de los gradientes.
    optimizador = torch.optim.Adam(modelo.parameters(), lr=TASA_APRENDIZAJE)

    # numel() cuenta todos los pesos y sesgos de cada tensor entrenable.
    print(f"Parámetros entrenables: {sum(p.numel() for p in modelo.parameters()):,}")

    for epoca in range(1, EPOCAS + 1):
        modelo.train()  # Activa el modo entrenamiento.
        perdida_acumulada = 0.0

        for imagenes, etiquetas in cargador_entrenamiento:
            optimizador.zero_grad()             # Elimina gradientes del lote anterior.
            logits = modelo(imagenes)            # Forward: calcula las puntuaciones.
            valor_perdida = perdida(logits, etiquetas)  # Calcula la pérdida.
            valor_perdida.backward()             # Backpropagation: calcula gradientes.
            optimizador.step()                   # Actualiza pesos y sesgos.

            # CrossEntropyLoss devuelve la pérdida media del lote por defecto.
            # La ponderamos por el número de imágenes para obtener la media global.
            perdida_acumulada += valor_perdida.item() * etiquetas.size(0)

        # Se evalúa el test después de cada época para informar del progreso.
        # Para escoger hiperparámetros rigurosamente, sería preferible usar
        # validación y reservar el test para la evaluación final.
        precision = evaluar(modelo, cargador_prueba)
        print(
            f"Época {epoca}/{EPOCAS} | "
            f"Pérdida train: {perdida_acumulada / len(entrenamiento):.4f} | "
            f"Acierto test: {precision:.2f}%"
        )

    # Guarda únicamente los parámetros aprendidos, no todo el objeto del modelo.
    # Los scripts de FGSM deben reconstruir la misma arquitectura para cargarlos.
    archivo = CARPETA / "cnn_mnist.pth"
    torch.save(modelo.state_dict(), archivo)
    print(f"Pesos guardados en: {archivo}")
    print("Objetivo mínimo del enunciado: 95% de acierto en MNIST.")


# Evita que se vuelva a entrenar al importar CNNMNIST desde otro archivo.
if __name__ == "__main__":
    main()
