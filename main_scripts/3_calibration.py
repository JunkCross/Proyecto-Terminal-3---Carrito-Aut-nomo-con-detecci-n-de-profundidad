import os
import cv2
import numpy as np
import json
from stereovision.calibration import StereoCalibrator
from stereovision.calibration import StereoCalibration
from stereovision.exceptions import ChessboardNotFoundError

# Variables globales preestablecidas
total_photos = 50

# Parámetros del tablero de ajedrez
# Debe utilizar un tablero de ajedrez de 6 filas y 9 columnas.
rows = 6
columns = 9
square_size = 2.6

image_size = (640, 480)

#Esta es la clase de calibración del paquete StereoVision
calibrator = StereoCalibrator(rows, columns, square_size, image_size)
photo_counter = 0
print('Iniciar ciclo')

#While bucle para la calibración. Revisará cada par de imágenes una por una.
while photo_counter != total_photos:
    photo_counter += 1 
    print('Importar par: ' + str(photo_counter))
    leftName = '/home/jetson/Documents/StereoVision_HOME_CUDA/calibracion/pairs/left_' + str(photo_counter).zfill(2) + '.png'
    rightName = '/home/jetson/Documents/StereoVision_HOME_CUDA/calibracion/pairs/right_' + str(photo_counter).zfill(2) + '.png'
    if os.path.isfile(leftName) and os.path.isfile(rightName):
        #leyendo las imágenes en color
        imgLeft = cv2.imread(leftName, 1)
        imgRight = cv2.imread(rightName, 1)

        #Asegurarse de que las imágenes izquierda y derecha tengan las mismas dimensiones
        (H, W, C) = imgLeft.shape

        imgRight = cv2.resize(imgRight, (W, H))

        # Calibrar la cámara (obtener las esquinas y dibujarlas)
        try:
            calibrator._get_corners(imgLeft)
            calibrator._get_corners(imgRight)
        except ChessboardNotFoundError as error:
            print(error)
            print("Par No " + str(photo_counter) + " ignored")
        else:
            #La función add_corners de Class ya nos ayuda con cv2.imshow,
            #y por lo tanto no necesitamos hacerlo por separado
            calibrator.add_corners((imgLeft, imgRight), True)
        
    else:
        print ("Par no encontrado")
        continue


print('¡Ciclo completo!')

print('Iniciando la calibración... ¡Puede tardar varios minutos!')
calibration = calibrator.calibrate_cameras()
calibration.export('/home/jetson/Documents/StereoVision_HOME_CUDA/calibracion/calib_result')
print('Calibration complete!')

# Rectifiquemos y mostremos el último par después de la calibración.
calibration = StereoCalibration(input_folder='/home/jetson/Documents/StereoVision_HOME_CUDA/calibracion/calib_result')
rectified_pair = calibration.rectify((imgLeft, imgRight))

cv2.imshow('Izquierda Calibrada!', rectified_pair[0])
cv2.imshow('Derecha Calibrada!', rectified_pair[1])

cv2.imwrite("/home/jetson/Documents/StereoVision_HOME_CUDA/calibracion/rectified_left.jpg", rectified_pair[0])
cv2.imwrite("/home/jetson/Documents/StereoVision_HOME_CUDA/calibracion/rectified_right.jpg", rectified_pair[1])
cv2.waitKey(0)