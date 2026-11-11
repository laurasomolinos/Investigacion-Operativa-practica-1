"""Visualización en 2D del clasificador L1-SVM y de sus márgenes."""
import numpy as np
import matplotlib.pyplot as plt

TOL = 1e-9


def dibujar_clasificador(X, y, w, intercepto, alpha, precision):
    """Dibuja puntos, frontera de decisión, márgenes y vectores soporte."""
    # Representación de las clases, los vectores soporte y los márgenes.
    plt.figure(figsize=(9, 6))
    plt.scatter(X[y == -1, 0], X[y == -1, 1], c='tab:blue', label='Clase -1', s=45)
    plt.scatter(X[y == 1, 0], X[y == 1, 1], c='tab:orange', label='Clase +1', s=45)

    # En el dual, los puntos con alpha positivo son vectores soporte.
    soportes = alpha > 1e-7
    plt.scatter(X[soportes, 0], X[soportes, 1], s=170,
                facecolors='none', edgecolors='black', linewidths=1.7,
                label='Vectores soporte')

    # Dibujamos la frontera de decisión f(x)=0 y los márgenes f(x)=+1 y -1.
    xx = np.linspace(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, 200)
    if abs(w[1]) > TOL:
        for nivel, estilo, etiqueta in [(0, '-', 'Separador f(x)=0'),
                                       (1, '--', 'Márgenes f(x)=±1'),
                                       (-1, '--', None)]:
            yy = (nivel - w[0] * xx - intercepto) / w[1]
            plt.plot(xx, yy, linestyle=estilo, color='black',
                     linewidth=1.8 if nivel == 0 else 1.1, label=etiqueta)
    else:
        # Cuando w2=0, tanto el separador como los márgenes son verticales.
        for nivel, estilo, etiqueta in [(0, '-', 'Separador f(x)=0'),
                                       (1, '--', 'Márgenes f(x)=±1'),
                                       (-1, '--', None)]:
            posicion = (nivel - intercepto) / w[0]
            plt.axvline(posicion, linestyle=estilo, color='black',
                        linewidth=1.8 if nivel == 0 else 1.1, label=etiqueta)

    plt.xlabel('Característica 1')
    plt.ylabel('Característica 2')
    plt.title(f'L1-SVM resuelto mediante Símplex — precisión: {100*precision:.0f}%')
    plt.legend(loc='best')
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.show()
