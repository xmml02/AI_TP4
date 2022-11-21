import cv2
import numpy as npy

'''
Nota importante:

La libreria openCV (cv2) contiene distintos metodos, entre ellos el de Hough, para estimar patrones.
Este prototipo utiliza la librería solamente en los siguientes casos:
    - Generar la matriz con los pixeles de la imagen a tratar
    - Generar imagen con la originan con patrones superpuestos
'''

'''
Primer Paso - Generar Matriz de imagen
Este metodo simplifica la imagen y sus colores.
El objetivo es que el algoritmo trabaje sobre las coordenadas que pretendemos analizar y buscar patrones.
Asimismo, la matriz nos provee de las dimensiones y los parametros para dibujar nuestro objeto en el eje de abscisas
'''
imagenInput = cv2.imread('TP4_circ.jpg')
imagenInput = cv2.medianBlur(imagenInput,5)

# Declarar variables con las dimensiones
dimY = imagenInput.shape[0]
dimX = imagenInput.shape[1]

# Con las dimensiones determinamos la diagonal
distanciaMax = int(npy.round(npy.sqrt(dimX ** 2 + dimY ** 2)))

'''
Segundo paso - Generar Modelo Parametrico
Aquí generamos un array que determinara la cantidad de votaciones
'''
# Theta in range from -90 to 90 degrees
arrayThetas = npy.deg2rad(npy.arange(-90, 90))

# Range of radius
rangoRadios = npy.linspace(-distanciaMax, distanciaMax, 2 * distanciaMax)

# instanciamos el acumulador calcular las votaciones
radio = 120
acumulador = npy.zeros((2 * distanciaMax, len(arrayThetas)))

'''
Tercer paso - Recorrer cada coordenada de nuestro espacio representado en una matriz
'''
for y in range(dimY):  # eje Y
    for x in range(dimX):  # eje X

        # Se verifican los puntos con el valor [0,0,0]
        temp = imagenInput[y, x]

        if temp.sum() == 0:  # la suma de los colores igual a 0

            #for r in range(radio):
                for n in range(0,361):

                    # Se recorre cada celda de la discretizacion del espacio de Hough
                    for k in range(len(arrayThetas)):
                        b = y - radio * npy.sin(n)
                        a = x - radio * npy.cos(n)

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
# r2=(x−x0)2+(y−y0)2

a = npy.cos(theta)
b = npy.sin(theta)
x0 = (a * rho) + 1000
y0 = (b * rho) + 1000

'''x1 = x0 + 1000 * -b
y1 = y0 + 1000 * a

x2 = x0 - 1000 * -b
y2 = y0 - 1000 * a'''

cv2.circle(imagenInput, (150, 150), 120, (0, 0, 255), 2)
# cv2.circle(imagenInput, (x0, y0), 2, (0, 0, 255), 2)

# cv2.line(imagenInput, (x1, y1), (x2, y2), (0, 0, 255), 2)

cv2.imwrite('salida_Circulo.jpg', imagenInput)
