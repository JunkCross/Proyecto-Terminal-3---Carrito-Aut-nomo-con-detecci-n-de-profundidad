import cv2
import numpy as np
from start_cameras import Start_Cameras
from datetime import datetime
import time
import os
from os import path

#Preajustes de toma de fotos
total_photos = 50  # Número de imágenes a tomar
countdown = 10  # Intervalo para el temporizador de cuenta regresiva, segundos
font = cv2.FONT_HERSHEY_SIMPLEX  # Fuente del temporizador de cuenta regresiva


def TakePictures():
    val = input("¿Le gustaría comenzar la captura de imagenes? (Y/N) ")

    if val.lower() == "y":
        left_camera = Start_Cameras(1).start()
        right_camera = Start_Cameras(0).start()
        cv2.namedWindow("Images", cv2.WINDOW_NORMAL)

        counter = 0
        t2 = datetime.now()
        while counter <= total_photos:
            #configurar la cuenta regresiva
            t1 = datetime.now()
            countdown_timer = countdown - int((t1 - t2).total_seconds())

            left_grabbed, left_frame = left_camera.read()
            right_grabbed, right_frame = right_camera.read()

            if left_grabbed and right_grabbed:
                #combina las dos imágenes juntas
                images = np.hstack((left_frame, right_frame))
                #guarda las imágenes una vez que se acabe la cuenta regresiva
                if countdown_timer == -1:
                    counter += 1
                    print(counter)

                 #Compruebe si el directorio existe. Guarde la imagen si existe. Cree una carpeta y luego guarde las imágenes si no es así.
                    if path.isdir('/home/jetson/Documents/StereoVision_HOME_CUDA/calibracion/images') == True:
                        #zfill(2) se utiliza para garantizar que siempre haya 2 dígitos, por ejemplo, 01/02/11/12
                        filename = "/home/jetson/Documents/StereoVision_HOME_CUDA/calibracion/images/image_" + str(counter).zfill(2) + ".png"
                        cv2.imwrite(filename, images)
                        print("Image: " + filename + " esta guardada!")
                    else:
                        #Haciendo directorio
                        os.makedirs("/home/jetson/Documents/StereoVision_HOME_CUDA/calibracion/images")
                        filename = "/home/jetson/Documents/StereoVision_HOME_CUDA/calibracion/images/image_" + str(counter).zfill(2) + ".png"
                        cv2.imwrite(filename, images)
                        print("Image: " + filename + " esta guardada!")

                    t2 = datetime.now()
                    #suspende la ejecución por unos segundos
                    time.sleep(1)
                    countdown_timer = 0
                    next
                    
                # Agregar el temporizador de cuenta regresiva en las imágenes y mostrar las imágenes
                cv2.putText(images, str(countdown_timer), (50, 50), font, 2.0, (0, 0, 255), 4, cv2.LINE_AA)
                cv2.imshow("Images", images)

                k = cv2.waitKey(1) & 0xFF

                if k == ord('q'):
                    break
                    
            else:
                break

    elif val.lower() == "n":
        print("Quitting! ")
        exit()
    else:
        print ("Inténtalo de nuevo! ")

    left_camera.stop()
    left_camera.release()
    right_camera.stop()
    right_camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    TakePictures()
                
                
                
