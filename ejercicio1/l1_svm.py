"""Formulación matricial primal y dual de un L1-SVM lineal.

Primal: min ||w||_1 + C * suma(xi_i)
         s.a. y_i (w^T x_i + b) >= 1 - xi_i, xi_i >= 0.

Dual:   max suma(alpha_i)
         s.a. -1 <= suma(y_i x_ij alpha_i) <= 1,
              suma(y_i alpha_i) = 0,
              0 <= alpha_i <= C.

Ambos se devuelven como (c, A, b) con restricciones A x <= b, x >= 0,
que es la entrada del Símplex.
"""
import numpy as np


def construir_dual_svm(X, y, C=10.0):
    """Dual del L1-SVM: max suma(alpha) s.a. A alpha <= b, alpha >= 0.

    Filas de A (en este orden):
      d filas   YX^T alpha <= 1      } juntas: |suma y_i x_ij alpha_i| <= 1
      d filas  -YX^T alpha <= 1      }
      1 fila    y^T alpha <= 0       } juntas: suma y_i alpha_i = 0
      1 fila   -y^T alpha <= 0       }
      n filas   alpha <= C
    """
    n, d = X.shape
    YX = y[:, None] * X  # fila i = y_i * x_i
    A = np.vstack((YX.T, -YX.T, y[None, :], -y[None, :], np.eye(n)))
    b = np.r_[np.ones(2 * d), 0.0, 0.0, C * np.ones(n)]
    c = np.ones(n)  # objetivo: suma de las alphas
    return c, A, b


def recuperar_clasificador(multiplicadores, d, n):
    """Obtiene w, b y xi a partir de c_B B^-1 del óptimo del dual SVM.

    Cada restricción del dual tiene como multiplicador una variable del
    primal, en el mismo orden que las filas de construir_dual_svm:
    w+, w-, b+, b-, xi.
    """
    u = multiplicadores
    w = u[:d] - u[d:2*d]                 # w = w+ - w-
    intercepto = u[2*d] - u[2*d + 1]     # b = b+ - b-
    xi = u[2*d + 2:2*d + 2 + n]          # holguras del margen de cada punto
    return w, intercepto, xi


def construir_primal_svm(X, y, C=10.0):
    """Primal del L1-SVM: min c^T v s.a. A v <= b, v >= 0.

    Variables v = (w+, w-, b+, b-, xi), todas no negativas.
    Cada restricción y_i(w^T x_i + b) >= 1 - xi_i se multiplica por -1:
      -y_i x_i^T w+ + y_i x_i^T w- - y_i b+ + y_i b- - xi_i <= -1.
    """
    n, d = X.shape
    A = np.zeros((n, 2 * d + 2 + n))
    A[:, :d] = -y[:, None] * X
    A[:, d:2*d] = y[:, None] * X
    A[:, 2*d] = -y
    A[:, 2*d + 1] = y
    A[:, 2*d + 2:] = -np.eye(n)
    b = -np.ones(n)
    # Objetivo: suma(w+) + suma(w-) + C suma(xi)  (b+ y b- no cuestan).
    c = np.r_[np.ones(2 * d), 0.0, 0.0, C * np.ones(n)]
    return c, A, b
