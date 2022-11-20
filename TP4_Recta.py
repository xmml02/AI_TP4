import cv2
import numpy as npy

'''
Nota importante:

La libreria openCV (cv2) contiene distintos metodos, entre ellos el de Hough, para estimar patrones.
Este prototipo utiliza la librería solamente en los siguientes casos:
    - Generar la matriz con los pixeles de la imagen a tratar
    - Generar imagen con la original y los patrones superpuestos
'''

'''
Primer Paso - Generar Matriz de imagen
Este método simplifica la imagen y sus colores. 
El objetivo es que el algoritmo trabaje sobre las coordenadas que pretendemos analizar y buscar patrones. 
Asimismo, la matriz nos provee de las dimensiones y los parámetros para dibujar nuestro objeto en el eje de coordenadas.
'''
imagenInput = cv2.imread('TP4.png')

# Declarar variables con las dimensiones
dimY = imagenInput.shape[0]
dimX = imagenInput.shape[1]

# Con las dimensiones determinamos la diagonal
distanciaMax = int(npy.round(npy.sqrt(dimX ** 2 + dimY ** 2)))

'''
Segundo paso - Generar Modelo Paramétrico
Aquí generamos un array que determinará las dimensiones del acumulador
'''
arrayThetas = npy.deg2rad(npy.arange(-90, 90))

# Range of radius
rangoRadios = npy.linspace(-distanciaMax, distanciaMax, 2 * distanciaMax)
# instanciamos el acumulador para calcular las votaciones
acumulador = npy.zeros((2 * distanciaMax, len(arrayThetas)))


'''
Tercer paso - Recorrer cada coordenada de nuestro espacio representado en una matriz
Se itera cada coordenada de color negro (RGB [0,0,0]).
Cada punto genera la funcion paramétrica en el espacio de Hough. 
Las celdas que visita dicha función incrementan el acumulador en 1.
'''
for y in range(dimY):       # eje Y
    for x in range(dimX):   # eje X

        # Se verifican los puntos NEGROS
        temp = imagenInput[y, x]

        if temp.sum() == 0:     # la suma de los colores igual a 0

            # Se recorre cada celda de la discretizacion del espacio de Hough
            for k in range(len(arrayThetas)):

                # Se calcula a d (la distancia entre el origen y la curva) como la función con theta (el angulo)
                d = x * npy.cos(arrayThetas[k]) + y * npy.sin(arrayThetas[k])

                # Se actualiza el acumulador, es el que contiene la votación
                acumulador[int(d) + distanciaMax, k] += 1

'''
Cuarto Paso - En el espacio de Hough encontrar la celda mas votada 

Esto nos permite despejar los parametros para trazar la recta
'''
# Se busca la celda con mas votos en el acumulador
idx = npy.argmax(acumulador)
rho = rangoRadios[int(idx / acumulador.shape[1])]
theta = arrayThetas[idx % acumulador.shape[1]]

# funcion mediante, encontraremos dos puntos para incluirlos en el procedimiento cv2.line
a = npy.cos(theta)
b = npy.sin(theta)
x0 = a * rho
y0 = b * rho

x1 = int(x0 + 1000 * -b)
y1 = int(y0 + 1000 * a)

x2 = int(x0 - 1000 * -b)
y2 = int(y0 - 1000 * a)

cv2.line(imagenInput, (x1, y1), (x2, y2), (0, 0, 255), 2)

cv2.imwrite('TP4_Salida.jpg', imagenInput)
