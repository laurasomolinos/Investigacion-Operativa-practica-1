"""Punto de entrada del ejercicio 1 de Investigación Operativa.

Ejecutar desde esta carpeta con: python main.py
"""
import numpy as np

from datos import generar_datos
from simplex import simplex
from l1_svm import construir_dual_svm, construir_primal_svm, recuperar_clasificador
from graficas import dibujar_clasificador


def main():
    # Generamos dos categorías separables en dos dimensiones.
    X, y = generar_datos()
    n, d = X.shape
    C = 10.0

    # Formulamos el dual como PL y lo resolvemos con nuestro Símplex,
    # imprimiendo todas las iteraciones.
    print('=== Dual del L1-SVM ===')
    c, A, b = construir_dual_svm(X, y, C)
    res_dual = simplex(c, A, b, objetivo='max', verbose=True)
    alpha = res_dual['x']

    # c_B B^-1 en el óptimo del dual contiene las variables del primal:
    # de ahí se obtienen directamente w, b y xi.
    w, intercepto, xi = recuperar_clasificador(res_dual['multiplicadores'], d, n)

    # Comprobación: el primal (que arranca con Símplex dual) da el mismo valor.
    cp, Ap, bp = construir_primal_svm(X, y, C)
    res_primal = simplex(cp, Ap, bp, objetivo='min')

    # Mostramos resultados y verificamos la dualidad fuerte.
    predicciones = np.where(X @ w + intercepto >= 0, 1, -1)
    precision = np.mean(predicciones == y)
    print('\n=== Resultados ===')
    print('Valor óptimo dual:', round(res_dual['valor'], 6))
    print('Valor óptimo primal:', round(res_primal['valor'], 6))
    print('Diferencia primal-dual:', abs(res_dual['valor'] - res_primal['valor']))
    print('w:', np.round(w, 6), 'b:', round(intercepto, 6))
    print('Multiplicadores alpha no nulos:', np.count_nonzero(alpha > 1e-7))
    print('Precisión de clasificación:', f'{100 * precision:.1f}%')
    print('Máxima holgura:', round(float(np.max(xi)), 8))

    # Representación de la solución obtenida.
    dibujar_clasificador(X, y, w, intercepto, alpha, precision)


if __name__ == '__main__':
    main()
