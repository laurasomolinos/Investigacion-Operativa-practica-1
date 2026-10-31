## PRÁCTICA 1 INVESTIGACIÓN OPERATIVA

Laura Somolinos, Alejandro Blanco, Adriana González, Lucas Tianming Miguel

## PARTE 2:  realización de un ataque adversario a una red neuronal. 

Los cuatro primeros ítems son obligatorios y sólo con ellos se puede obtener un
10 en esta parte 2 de la práctica uno. Sin embargo, hay una serie de puntos opcionales para
aquellos tengan interés en probar más cosas. Se recomienda usar pytorch o tensorflow para
la realización de esta parte. El proceso es el siguiente:
1. Entrena una red neuronal que clasifique con al menos un 95% de acierto el conjunto de
datos de MNIST. Elige una arquitectura diferente a la de clase, y asegúrate de que no
coincida con la de ningún otro grupo. Puedes utilizar para esto todas las librerías que
desees. Si se usa tensorflow, se recomienda usar el dataset importado desde la propia
librería con from tensorflow.keras.datasets import mnist. Identifica la cantidad
de parámetros de la arquitectura de tu red neuronal.
2. Entrena una SVM con núcleo Gaussiano para MNIST. Puedes utilizar para esto todas
las librerías que desees.
3. Genera una función que haga un ataque de tipo FGSM a la red neuronal. Si existe una
librería de python específica para esto, no se puede usar, pero sí se puede usar (y es
recomendable) usar funciones que te ayuden a para calcular los gradientes dentro de la
función signo, (por ejemplo la función GradientTape de tensorflow).
4. Genera unas cuantas imágenes adversarias y trata de predecirlas tanto con tu red neuronal
como con la SVM. Visualiza tanto las originales como las adversarias.
5. (Opcional) Genera imágenes adversarias para todo el conjunto de train y/o test de MNIST
y establece el error de cada uno de los dos modelos sobre este conjunto adversario.
6. (Opcional) Entrena adversariamente, es decir, entrena tus dos modelos con las 120.000
imágenes; 60.000 originales y 60.000 adversarias. Calcula el error en el test conjunto
(20.000 imágenes, originales y adversarias).
Curso 26-27
Tema 5. Prácticas 99
7. (Opcional) Ataca a las 60.000 imágenes adversarias, obteniendo 60.000 imágenes “doblemente adversarias”. ¿Qué rendimiento obtiene el modelo entrenado adversariamente sobre
las imágenes “doblemente adversarias”