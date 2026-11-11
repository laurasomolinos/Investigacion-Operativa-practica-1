"""Apartado 3: ataque FGSM (Fast Gradient Sign Method) implementado a mano para la CNN de MNIST.

Ejecutar: python fgsm.py     (genera ejemplos_fgsm.png con 6 dígitos originales y adversarios)

FGSM (Goodfellow et al., 2015) modifica cada píxel un máximo de epsilon en la dirección que
MÁS AUMENTA la pérdida del modelo:

    x_adv = clip( x + epsilon * sign( d L(f(x), y) / d x ), 0, 1 )

- Solo se usa PyTorch para calcular el gradiente (torch.autograd.grad); el signo, la
  perturbación y el recorte están escritos aquí, sin librerías de ataques adversarios.
- La perturbación está acotada en norma L-infinito: ningún píxel cambia más de epsilon.
- Como solo importa el SIGNO del gradiente, que la pérdida sea la media del lote no altera el
  resultado (dividir por el tamaño del lote no cambia el signo).
"""
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn
from torchvision import datasets, transforms

from entrenar_cnn import CNNMNIST

CARPETA = Path(__file__).resolve().parent #carpeta del proyecto 
SEMILLA = 42
EPSILON = 0.10 #máxima perturbacion a permitir por pixel, probaremos distintos valores 
NUM_EJEMPLOS = 6 #6 ejemplos para visualizar el ataque


def ataque_fgsm(modelo: nn.Module, imagenes: torch.Tensor,
                etiquetas: torch.Tensor, epsilon: float) -> torch.Tensor:
    """Genera imágenes adversarias FGSM para aumentar el error de clasificación
    - modelo: red neuronal entrenada (devuelve logits)
    - imagenes: tensor de tamaño [lote, 1, 28, 28], con píxeles en [0, 1]
    - etiquetas: clases verdaderas, tensor de tamaño [lote]
    - epsilon: máxima variación absoluta permitida por píxel
    Returns:
    Tensor adversario, desconectado del grafo y con valores en [0, 1]
    """
    if not 0 <= epsilon <= 1: #perturbación del ataque dentro del margen 
        raise ValueError("epsilon debe estar entre 0 y 1 para imágenes normalizadas")

    #modo evaluación: desactiva dropout y batchnorm para que no se comporten de forma diferente a como lo hicieron durante el entrenamiento
    modelo.eval()
    #creamos una copia independiente de las imágenes y activamos el cálculo de gradientes respecto a sus píxeles, necesarios para generar FGSM.
    entrada = imagenes.detach().clone().requires_grad_(True)
    logits = modelo(entrada) #forward pass
    perdida = nn.functional.cross_entropy(logits, etiquetas) #perdida 
    #calculamos únicamente el gradiente respecto a las imágenes, sin acumular gradientes ni actualizar los pesos de la CNN
    gradiente = torch.autograd.grad(perdida, entrada)[0]

    #FGSM: perturbamos cada píxel en la dirección que aumenta la pérdida
    adversarias = entrada + epsilon * gradiente.sign()
    #los píxeles deben seguir siendo imágenes válidas [0, 1]
    adversarias = adversarias.clamp(0.0, 1.0)
    return adversarias.detach() #detach() para separar el tensor y que no se sigan calculando gradientes


def main() -> None:
    torch.manual_seed(SEMILLA)
    ruta_pesos = CARPETA / "cnn_mnist.pth"#cargar pesos de CNN entrenada 
    if not ruta_pesos.exists():
        raise FileNotFoundError(f"No encuentro {ruta_pesos}. Conserva el modelo del apartado 1.")

    modelo = CNNMNIST()
    #cargamos el state_dict: asigna los parámetros guardados a una CNN con la misma arquitectura
    modelo.load_state_dict(torch.load(ruta_pesos, map_location="cpu", weights_only=True))
    modelo.eval()

    #10.000 imágenes de test
    prueba = datasets.MNIST(root=CARPETA / "datos", train=False, download=True, transform=transforms.ToTensor())

    #6 ejemplos reproducibles, uno por una clase distinta, para visualizar.
    imagenes, etiquetas = [], []
    clases_elegidas = set()
    #termina al encontrar 6 ejemplos de distintas clases
    for imagen, etiqueta in prueba:
        if etiqueta not in clases_elegidas:
            imagenes.append(imagen)
            etiquetas.append(etiqueta)
            clases_elegidas.add(etiqueta)
        if len(imagenes) == NUM_EJEMPLOS:
            break

    x = torch.stack(imagenes) #agrupa 6 imagenes en un tensor de 6 (1,28,28) -> (6,1,28,28)
    y = torch.tensor(etiquetas, dtype=torch.long) #etiquetas a tensor
    x_adv = ataque_fgsm(modelo, x, y, EPSILON) #ejecuta ataque 

    with torch.no_grad(): #comparar predicciones 
        pred_original = modelo(x).argmax(dim=1)
        pred_adversaria = modelo(x_adv).argmax(dim=1)

    #mostrar resultados 
    print(f"FGSM manual | epsilon={EPSILON} | ejemplos={len(y)}")
    print("Etiqueta real | Predicción original | Predicción adversaria")
    for real, antes, despues in zip(y.tolist(), pred_original.tolist(), pred_adversaria.tolist()):
        print(f"{real:^13} | {antes:^19} | {despues:^21}")

    perturbacion_max = (x_adv - x).abs().max().item()
    print(f"Variación máxima real de un píxel: {perturbacion_max:.5f}")
    assert perturbacion_max <= EPSILON + 1e-6

    #mostrar imágenes
    fig, ejes = plt.subplots(2, len(y), figsize=(2.5 * len(y), 5))
    for i in range(len(y)):
        ejes[0, i].imshow(x[i, 0].numpy(), cmap="gray", vmin=0, vmax=1)
        ejes[0, i].set_title(f"Original: {pred_original[i].item()}")
        ejes[1, i].imshow(x_adv[i, 0].numpy(), cmap="gray", vmin=0, vmax=1)
        ejes[1, i].set_title(f"FGSM: {pred_adversaria[i].item()}")
        for fila in range(2):
            ejes[fila, i].axis("off")
    fig.suptitle(f"Ataque FGSM sobre MNIST (epsilon={EPSILON})")
    fig.tight_layout()
    destino = CARPETA / "ejemplos_fgsm.png"
    fig.savefig(destino, dpi=160)
    plt.close(fig)
    print(f"Comparación guardada en: {destino}")


if __name__ == "__main__":
    main()
