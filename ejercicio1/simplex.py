"""Algoritmo Símplex combinado (primal + dual) en forma matricial.

Resuelve
    max (o min) c^T x
    sujeto a    A x <= b,  x >= 0
introduciendo una variable de holgura s_i por cada restricción.
Las restricciones >= se introducen multiplicadas por -1.

Pasos (según los apuntes de clase):
  1. Si x = 0 es factible (todos los b_i >= 0) se empieza con Símplex primal.
  2. Si no, se empieza con Símplex dual:
       sale la básica con b_i más negativo,
       entra la variable con mínimo |z_j / a_sj| entre las a_sj < 0.
  3. Cuando todos los b_i >= 0 se cambia a Símplex primal:
       entra la variable con z_j más negativo,
       sale la básica con mínimo b_i / a_ie entre las a_ie > 0.
  4. Es óptimo cuando todo el renglón z = c_B B^-1 A - c es >= 0.
En cada iteración se multiplica la matriz original por
    [ 1  c_B B^-1 ]
    [ 0    B^-1   ].
No se tratan casos degenerados.
"""
import numpy as np

TOL = 1e-9
MAX_ITER = 1000


def imprimir_matriz(M, filas, columnas):
    """Imprime una matriz con etiquetas de filas y columnas."""
    ancho = 8
    print(' ' * 6 + ''.join(f'{col:>{ancho}}' for col in columnas))
    for nombre, fila in zip(filas, M):
        valores = ''.join(f'{0.0 if abs(v) < TOL else v:>{ancho}.3g}' for v in fila)
        print(f'{nombre:>6}' + valores)


def simplex(c, A, b, objetivo='max', verbose=False):
    """Resuelve max/min c^T x s.a. Ax <= b, x >= 0 con el Símplex combinado.

    Devuelve un diccionario con:
      x                solución (variables originales)
      valor            valor óptimo de la función objetivo
      multiplicadores  c_B B^-1 en el óptimo (solución del problema dual)
      base             nombres de las variables básicas finales
      historial        lista con la base y el punto de cada iteración
    """
    c = np.asarray(c, dtype=float)
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    m, n = A.shape

    # Trabajamos siempre maximizando: min c^T x  =  -max (-c)^T x.
    signo = 1.0 if objetivo == 'max' else -1.0

    # Matriz ampliada con las holguras: [A | I] y costes [c | 0].
    A_amp = np.hstack((A, np.eye(m)))
    costes = np.r_[signo * c, np.zeros(m)]
    nombres = [f'x{j + 1}' for j in range(n)] + [f's{i + 1}' for i in range(m)]

    # Base inicial: las holguras (su matriz B es la identidad).
    base = list(range(n, n + m))

    # Paso 1: si x = 0 es factible, primal; si no, dual.
    metodo = 'primal' if np.all(b >= -TOL) else 'dual'

    historial = []
    entra = sale = None
    for k in range(MAX_ITER):
        # Matrices de la iteración.
        B_inv = np.linalg.inv(A_amp[:, base])
        cB_Binv = costes[base] @ B_inv        # c_B B^-1
        xB = B_inv @ b                        # B^-1 b (columna de la derecha)
        tabla = B_inv @ A_amp                 # B^-1 A
        z = cB_Binv @ A_amp - costes          # renglón z
        valor = cB_Binv @ b                   # valor actual del objetivo

        # Punto actual en las variables originales (para dibujar y tabular).
        x = np.zeros(n + m)
        x[base] = xB

        # Paso 3: cuando la solución ya es factible se cambia a primal.
        if metodo == 'dual' and np.all(xB >= -TOL):
            metodo = 'primal'
        optimo = metodo == 'primal' and np.all(z >= -TOL)

        historial.append({'iteracion': k, 'metodo': 'óptimo' if optimo else metodo,
                          'base': [nombres[j] for j in base], 'x': x[:n].copy(),
                          'valor': float(signo * valor) + 0.0,  # + 0.0 evita el -0.0
                          'entra': entra, 'sale': sale})

        if verbose:
            if k == 0:
                print('Base inicial:', [nombres[j] for j in base])
            else:
                print(f'\nIteración {k}')
                print(f'  Sale:  {sale}')
                print(f'  Entra: {entra}')
                print('  Nueva base:', [nombres[j] for j in base])
            # Matriz [[1, c_B B^-1], [0, B^-1]].
            P = np.block([[np.ones((1, 1)), cB_Binv[None, :]],
                          [np.zeros((m, 1)), B_inv]])
            print('\n  Matriz [1  c_B B^-1 ; 0  B^-1]:')
            imprimir_matriz(P, ['z'] + [f'f{i + 1}' for i in range(m)],
                            [''] + [f'f{i + 1}' for i in range(m)])
            # Tabla del Símplex: renglón z y B^-1 A | B^-1 b.
            print('\n  Tabla Símplex:')
            T = np.vstack((np.r_[z, valor], np.column_stack((tabla, xB))))
            imprimir_matriz(T, ['z'] + [nombres[j] for j in base], nombres + ['b'])

        if optimo:
            if verbose:
                print(f'\nÓptimo: valor = {signo * valor:.6g}')
            return {'x': x[:n], 'valor': float(signo * valor),
                    'multiplicadores': cB_Binv, 'base': [nombres[j] for j in base],
                    'historial': historial}

        if metodo == 'dual':
            # Paso 2 (Símplex dual).
            # Sale la básica con b_i más negativo.
            s = int(np.argmin(xB))
            # Entra la no básica con a_sj < 0 y mínimo |z_j / a_sj|.
            candidatas = [j for j in range(n + m) if j not in base and tabla[s, j] < -TOL]
            if not candidatas:
                raise ValueError('Problema infactible.')
            e = min(candidatas, key=lambda j: abs(z[j] / tabla[s, j]))
        else:
            # Paso 3 (Símplex primal).
            # Entra la variable con z_j más negativo.
            e = int(np.argmin(z))
            # Sale la básica con a_ie > 0 y mínimo b_i / a_ie.
            filas = [i for i in range(m) if tabla[i, e] > TOL]
            if not filas:
                raise ValueError('Problema no acotado.')
            s = min(filas, key=lambda i: xB[i] / tabla[i, e])

        sale, entra = nombres[base[s]], nombres[e]
        base[s] = e  # Cambio de base.
    raise RuntimeError('Se ha superado el número máximo de iteraciones.')
