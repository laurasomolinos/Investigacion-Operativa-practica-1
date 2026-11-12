""" apartados 4 a 7 - No ejecutado directamente, importado por experimentos.ipynb

Contiene funciones implementadas: 
- Carga de los datos MNIST.
- Generación de imágenes adversarias mediante FGSM.
- Predicción y evaluación de la CNN.
- Clase SVMTorch para utilizar la SVM con PyTorch y calcular gradientes.
- Entrenamiento de la CNN con imágenes originales y adversarias.
"""

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets

# Importamos la arquitectura de la CNN y la función que fija las semillas.
from entrenar_cnn import CNNMNIST, fijar_semillas
# Reutilizamos la implementación manual de FGSM del apartado 3.
from fgsm import ataque_fgsm


def cargar_mnist(carpeta: Path):
    """
    Carga los conjuntos de entrenamiento y test de MNIST.
    Devuelve las imágenes como tensores de PyTorch normalizados entre 0 y 1, junto con sus etiquetas.
    """
    #60000 imagenes de entrenamiento
    tr = datasets.MNIST(root=carpeta / "datos", train=True, download=True)
    
    #10000 imagenes de test
    te = datasets.MNIST(root=carpeta / "datos", train=False, download=True)
    #pixeles a decimales y normalizados entre 0 y 1, obtenemos tensores: [num_imagenes, 1, 28 28]
    X_tr = (tr.data.float() / 255.0).unsqueeze(1) # 
    X_te = (te.data.float() / 255.0).unsqueeze(1)
    #devuelve imagenes y etiquetas, long() convierte etiquetas a tipo entero 
    return X_tr, tr.targets.long(), X_te, te.targets.long()


def generar_adversarias(modelo: nn.Module, X: torch.Tensor, y: torch.Tensor,
                        epsilon: float, lote: int = 1000) -> torch.Tensor:
    #Genera imágenes adversarias para un conjunto completo utilizando FGSM. 
    #Se procesan las imágenes por lotes para reducir el consumo de memoria.
    salida = [ataque_fgsm(modelo, X[i:i + lote], y[i:i + lote], epsilon)
              for i in range(0, len(X), lote)]
    # Unimos todos los lotes para obtener un único tensor de imágenes adversarias.
    return torch.cat(salida)



#PREDICCIÓN Y EVALUACION CNN
#no se necesita calculo de gradientes para realizar predicciones 
@torch.no_grad()
def predecir_cnn(modelo: nn.Module, X: torch.Tensor, lote: int = 2000) -> np.ndarray:
    modelo.eval() #modo evaluacion 
    #dividir imagenes en lotes y obtener predicciones maximas 
    return torch.cat([modelo(X[i:i + lote]).argmax(1) for i in range(0, len(X), lote)]).numpy()


def precision(pred: np.ndarray, y) -> float:
    #porcentaje de predicciones correctas 
    y = y.numpy() if torch.is_tensor(y) else np.asarray(y)
    return 100.0 * float((pred == y).mean())



#REPRESENTACION DE SVM EN PYTORCH 
class SVMTorch(nn.Module):
    """
    Representación de una SVM RBF entrenada con scikit-learn dentro de PyTorch.
    Necesitamos esta clase porque queremos aplicar ataques FGSM directamente
    a la SVM y para ello necesitamos calcular gradientes con PyTorch.
    predict() -> reproduce la votación de la SVM original.
    forward() -> devuelve puntuaciones diferenciables por clase,
    que permiten calcular gradientes y aplicar FGSM.
    Estas puntuaciones son una aproximación diferenciable de la votación
    original, no exactamente la misma regla de clasificación.
    """

    def __init__(self, svm, dtype=torch.float32):
        super().__init__()
        # Guardamos el tipo de dato que utilizaremos en los tensores.
        self.dtype = dtype
        # Recuperamos el valor de gamma utilizado por la SVM entrenada.
        self.gamma = float(svm._gamma)
        # Recuperamos los vectores soporte de la SVM y los convertimosmen tensores de PyTorch.
        self.register_buffer("sv", torch.tensor(svm.support_vectors_, dtype=dtype))
        # Calculamos la norma cuadrática de cada vector soporte-> para calcular distancias.
        self.register_buffer("sv_norma", (self.sv ** 2).sum(1))
        n_clases = len(svm.classes_) #10 clases en MNIST
        #posiciones iniciales de los vectores soporte
        inicio = np.concatenate([[0], np.cumsum(svm.n_support_)])
        # Recuperamos los coeficientes de los vectores soporte
        dual = svm.dual_coef_

        #todas las parejas posibles de clases.
        pares = [(i, j) for i in range(n_clases) for j in range(i + 1, n_clases)]

        # Matriz que almacenará los coeficientes de los vectores soporte
        M = np.zeros((self.sv.shape[0], len(pares)))

        # Matriz que relaciona cada clasificador binario con sus dos clases-> para puentuacion por clase 
        S = np.zeros((len(pares), n_clases))

        #Recorremos los 45 clasificadores binarios.
        for p, (i, j) in enumerate(pares):
            #coeficientes correspondientes a los vectores soporte de la primera clase de la pareja.
            M[inicio[i]:inicio[i + 1], p] = (dual[j - 1, inicio[i]:inicio[i + 1]])
            #segunda clase
            M[inicio[j]:inicio[j + 1], p] = (dual[i, inicio[j]:inicio[j + 1]])
            S[p, i], S[p, j] = 1.0, -1.0
        # Guardamos matrices y los términos independientes de la SVM
        self.register_buffer("M", torch.tensor(M, dtype=dtype))
        self.register_buffer("b", torch.tensor(svm.intercept_, dtype=dtype))
        self.register_buffer("S", torch.tensor(S, dtype=dtype))

        # qué clase recibe el voto cuando la decisión el clasificador binario es positiva.
        self.register_buffer("gana_i", torch.tensor(np.maximum(S, 0), dtype=dtype))
        #cuando es negativa o cero
        self.register_buffer("gana_j", torch.tensor(np.maximum(-S, 0), dtype=dtype))


    #DECISIONES DE CLASIFICAION BINARIA 
    def decisiones(self, X: torch.Tensor) -> torch.Tensor:
        """
        Calcula el valor de decisión de los 45 clasificadores binarios para las imágenes que recibe como entrada.
        """
        # cada imagen de 28x28 en un vector de 784 píxeles.
        x = X.reshape(len(X), -1).to(self.dtype)
        #distancias euclídeas al cuadrado entre cada imagen y cada vector soporte.
        d2 = ((x ** 2).sum(1, keepdim=True) + self.sv_norma - 2 * x @ self.sv.T)
        #kernel RBF (gaussiano) utilizando las distancias
        K = torch.exp(-self.gamma * d2.clamp_min(0))

        #valores del kernel con los coeficientes de los clasificadores y sus términos independientes.
        # El resultado contiene 45 valores de decisión por imagen.
        return K @ self.M + self.b

    #PUNTUACIONES DIFERENCIABLES POR CLASE 
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.decisiones(X) @ self.S

    
    # Predicción mediante la votación original de la SVM
    @torch.no_grad()
    def predict(self, X: torch.Tensor, lote: int = 1000) -> np.ndarray:
        salida = []
        #procesar imagenes por lotes 
        for i in range(0, len(X), lote):
            #decisiones de los 45 clasificadores binarios.
            d = self.decisiones(X[i:i + lote])
            # Si la decisión es positiva, gana la primera clase.
            # Si es negativa o cero, gana la segunda clase.
            gana = (d > 0).to(self.dtype)
            #suma de los votos que recibe cada una de las 10 clases.
            votos = (gana @ self.gana_i + (1 - gana) @ self.gana_j)
            #clase que ha obtenido más votos.
            salida.append(votos.argmax(1))
        #predicciones de todos los lotes
        return torch.cat(salida).numpy()



#ENTRENAMIENTO DE LA CNN CON LOS DATOS PROPORCIONADOS

def entrenar_cnn(X: torch.Tensor, y: torch.Tensor, epocas: int = 4, lote: int = 128,
    lr: float = 1e-3, semilla: int = 42, X_val=None, y_val=None, verboso=True
):
    """
    Entrena la CNN utilizando las imágenes y etiquetas que recibe.
    Utiliza la misma arquitectura y el mismo procedimiento de entrenamiento que entrenar_cnn.py.
    Permite entrenar tanto con imágenes originales como con imágenes
    adversarias, dependiendo de los datos que se le proporcionen.
    """
    fijar_semillas(semilla)
    # generador aleatorio con semilla fija
    gen = torch.Generator().manual_seed(semilla)
    #conjunto de datos con las imágenes y sus etiquetas, divide los datos en lotes y los mezcla en cada época.
    datos = DataLoader(TensorDataset(X, y), batch_size=lote,
        shuffle=True, num_workers=0, generator=gen)
    #nueva CNN con la arquitectura del apartado 1.
    modelo = CNNMNIST()
    #Adam para actualizar los pesos de la red.
    opt = torch.optim.Adam(modelo.parameters(), lr=lr)
    #pérdida para clasificación multiclase.
    perdida = nn.CrossEntropyLoss()
    for ep in range(1, epocas + 1):
        #red en modo entrenamiento.
        modelo.train()
        #Acumulamod la pérdida para calcular la media de la época
        acumulada = 0.0
        for xb, yb in datos:
            #reiniciamos los gradientes acumulados 
            opt.zero_grad()
            v = perdida(modelo(xb), yb) #predcicciones y pérdida 
            # gradientes de la pérdida respecto a los pesos.
            v.backward()
            opt.step() #actuañizacion pesos de la red con Adam
            #acumular pérdida teniendo en cuenta el número de imágenes del lote, para calcular después la media de toda la época.
            acumulada += v.item() * len(yb)

        #si está activado el modo informativo, mostramos los resultados.
        if verboso:
            msg = (f"Época {ep}/{epocas} | "f"pérdida train {acumulada / len(X):.4f}")
            #porcentaje de acierto de conjunto de evaluacion.
            if X_val is not None:
                msg += (f" | acierto test limpio " f"{precision(predecir_cnn(modelo, X_val), y_val):.2f}%")
            print(msg)
    modelo.eval()
    return modelo
