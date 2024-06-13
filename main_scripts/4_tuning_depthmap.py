import cv2
import os
import threading
import numpy as np
import time
from datetime import datetime
import json
from stereovision.calibration import StereoCalibration
from start_cameras import Start_Cameras

# Función de mapa de profundidad
SWS = 215
PFS = 115
PFC = 43
MDS = -25
NOD = 112
TTH = 100
UR = 10
SR = 15
SPWS = 100

loading = False


def stereo_depth_map(rectified_pair, variable_mapping):

    '''print ('SWS='+str(SWS)+' PFS='+str(PFS)+' PFC='+str(PFC)+' MDS='+\
           str(MDS)+' NOD='+str(NOD)+' TTH='+str(TTH))
    print (' UR='+str(UR)+' SR='+str(SR)+' SPWS='+str(SPWS))'''
    #Prueba mitad de pixeles
    width = 640
    height = 480
    downsample = 2
    resize = (int(width/downsample), int(height/downsample)) 

    #blockSize es el SAD Window Size
    #Configuración del filtro
    sbm = cv2.StereoBM_create(numDisparities=64, blockSize=variable_mapping["SWS"]) 
    sbm.setPreFilterType(1)    
    sbm.setPreFilterSize(variable_mapping['PreFiltSize'])
    sbm.setPreFilterCap(variable_mapping['PreFiltCap'])
    sbm.setSpeckleRange(variable_mapping['SpeckleRange'])
    sbm.setSpeckleWindowSize(variable_mapping['SpeckleSize'])
    sbm.setMinDisparity(variable_mapping['MinDisp'])
    sbm.setNumDisparities(variable_mapping['NumofDisp'])
    sbm.setTextureThreshold(variable_mapping['TxtrThrshld'])
    sbm.setUniquenessRatio(variable_mapping['UniqRatio'])
    

    #c, r = rectified_pair[0].shape
    dmLeft = cv2.resize(rectified_pair[0], resize)
    dmRight = cv2.resize(rectified_pair[1], resize)
    #dmLeft = rectified_pair[0]
    #dmRight = rectified_pair[1]
    disparity = sbm.compute(dmLeft, dmRight)


    #//Esto aumenta el tamaño de la disparidad
    disparity = cv2.dilate(disparity, None, iterations=1)
    disparity = cv2.resize(disparity, (width, height))
    #//Esto aumenta el tamaño de la disparidad
    
    
    disparity_normalized = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
    #Convering Numpy Array to CV_8UC1
    image = np.array(disparity_normalized, dtype = np.uint8)
    disparity_color = cv2.applyColorMap(image, cv2.COLORMAP_JET)
    h, w, _ = disparity_color.shape
    #print("TAmaño de la imagen de disparity_color: ", (h, w))
    
    return disparity_color, disparity_normalized

def save_load_map_settings(current_save, current_load, variable_mapping):
    global loading
    if current_save != 0:
        print('Guardar en archivo...')

        result = json.dumps({'SADWindowSize':variable_mapping["SWS"], 'preFilterSize':variable_mapping['PreFiltSize'], 'preFilterCap':variable_mapping['PreFiltCap'], 
                'minDisparity':variable_mapping['MinDisp'], 'numberOfDisparities': variable_mapping['NumofDisp'], 'textureThreshold':variable_mapping['TxtrThrshld'], 
                'uniquenessRatio':variable_mapping['UniqRatio'], 'speckleRange':variable_mapping['SpeckleRange'], 'speckleWindowSize':variable_mapping['SpeckleSize']},
                sort_keys=True, indent=4, separators=(',',':'))
        fName = '/home/alan/Documentos/StereoVision_HOME_CUDA/calibracion/3dmap_set.txt'
        f = open (str(fName), 'w')
        f.write(result)
        f.close()
        print ('Configuración guardada en el archivo '+fName)


    if current_load != 0:
        if os.path.isfile('/home/alan/Documentos/StereoVision_HOME_CUDA/calibracion/3dmap_set.txt') == True:
            loading = True
            fName = '/home/alan/Documentos/StereoVision_HOME_CUDA/calibracion/3dmap_set.txt'
            print('Cargando parámetros desde el archivo...')
            f=open(fName, 'r')
            data = json.load(f)

            cv2.setTrackbarPos("SWS", "Stereo", data['SADWindowSize'])
            cv2.setTrackbarPos("PreFiltSize", "Stereo", data['preFilterSize'])
            cv2.setTrackbarPos("PreFiltCap", "Stereo", data['preFilterCap'])
            cv2.setTrackbarPos("MinDisp", "Stereo", data['minDisparity']+100)
            cv2.setTrackbarPos("NumofDisp", "Stereo", int(data['numberOfDisparities']/16))
            cv2.setTrackbarPos("TxtrThrshld", "Stereo", data['textureThreshold'])
            cv2.setTrackbarPos("UniqRatio", "Stereo", data['uniquenessRatio'])
            cv2.setTrackbarPos("SpeckleRange", "Stereo", data['speckleRange'])
            cv2.setTrackbarPos("SpeckleSize", "Stereo", data['speckleWindowSize'])

            f.close()
            print ('Parámetros cargados desde el archivo '+fName)
            print ('Redibujar el mapa de profundidad con parámetros cargados...')
            print ('Hecho!') 
        else: 
            print ("El archivo para cargar no existe.")

def activateTrackbars(x):
    global loading
    loading = False
    

def create_trackbars() :
    global loading

    #max_sws = min(640, 480)
    #SWS no puede ser mayor que el ancho y el alto de la imagen.
    #En este caso, ancho = 640 y alto = 480
    cv2.createTrackbar("SWS", "Stereo", 115, 230, activateTrackbars)
    cv2.createTrackbar("SpeckleSize", "Stereo", 0, 300, activateTrackbars)
    cv2.createTrackbar("SpeckleRange", "Stereo", 0, 40, activateTrackbars)
    cv2.createTrackbar("UniqRatio", "Stereo", 1, 20, activateTrackbars)
    cv2.createTrackbar("TxtrThrshld", "Stereo", 0, 1000, activateTrackbars)
    cv2.createTrackbar("NumofDisp", "Stereo", 1, 16, activateTrackbars)
    cv2.createTrackbar("MinDisp", "Stereo", -100, 200, activateTrackbars)
    cv2.createTrackbar("PreFiltCap", "Stereo", 1, 63, activateTrackbars)
    cv2.createTrackbar("PreFiltSize", "Stereo", 5, 255, activateTrackbars)
    cv2.createTrackbar("Save Settings", "Stereo", 0, 1, activateTrackbars)
    cv2.createTrackbar("Load Settings","Stereo", 0, 1, activateTrackbars)

def onMouse(event, x, y, flag, disparity_normalized):
    if event == cv2.EVENT_LBUTTONDOWN:
        distance = disparity_normalized[y][x]
        print("Distancia en centimetros {}".format(distance))



if __name__ == '__main__':
    left_camera = Start_Cameras(1).start()
    right_camera = Start_Cameras(0).start()

    try:
        # inicializamos trackbars and windows
        cv2.namedWindow("Stereo")
        create_trackbars()

        print ("Camaras iniciadas")

        variables = ["SWS", "SpeckleSize", "SpeckleRange", "UniqRatio", "TxtrThrshld", "NumofDisp",
        "MinDisp", "PreFiltCap", "PreFiltSize"]

        variable_mapping = {"SWS" : 15, "SpeckleSize" : 100, "SpeckleRange" : 15, "UniqRatio" : 10, "TxtrThrshld" : 100, "NumofDisp" : 1,
        "MinDisp": -25, "PreFiltCap" : 30, "PreFiltSize" : 105}

        while True:
            left_grabbed, left_frame = left_camera.read()
            right_grabbed, right_frame = right_camera.read()

            if left_grabbed and right_grabbed:
                left_gray_frame = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
                right_gray_frame = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)

                calibration = StereoCalibration(input_folder='/home/alan/Documentos/StereoVision_HOME_CUDA/calibracion/calib_result')
                rectified_pair = calibration.rectify((left_gray_frame, right_gray_frame))

                #obtener la posición de la barra de seguimiento y asignarla a las variables
                if loading == False:
                    for v in variables:
                        current_value = cv2.getTrackbarPos(v, "Stereo")
                        if v == "SWS" or v == "PreFiltSize":
                            if current_value < 5:
                                current_value = 5
                            if current_value % 2 == 0:
                                current_value += 1
                        
                        if v == "NumofDisp":
                            if current_value == 0:
                                current_value = 1
                            current_value = current_value * 16
                        if v == "MinDisp":
                            current_value = current_value - 100
                        if v == "UniqRatio" or v == "PreFiltCap":
                            if current_value == 0:
                                current_value = 1
                        
                        variable_mapping[v] = current_value

                
            #obteniendo guardar y cargar posiciones de la barra de seguimiento

                current_save = cv2.getTrackbarPos("Save Settings", "Stereo")
                current_load = cv2.getTrackbarPos("Load Settings", "Stereo")
    
                save_load_map_settings(current_save, current_load, variable_mapping)
                cv2.setTrackbarPos("Save Settings", "Stereo", 0)
                cv2.setTrackbarPos("Load Settings", "Stereo", 0)
                disparity_color, disparity_normalized = stereo_depth_map(rectified_pair, variable_mapping)

                # ¿Qué sucede cuando se hace clic con el mouse?
                cv2.setMouseCallback("Stereo", onMouse, disparity_normalized)
                        
                cv2.imshow("Stereo", disparity_color)
                cv2.imshow("Frame", np.hstack((rectified_pair[0], rectified_pair[1])))
                
                k = cv2.waitKey(1) & 0xFF
                if k == ord('q'):
                    break

                else:
                    continue
    except Exception as e:
        print("Ocurrió un error:", str(e))


    finally:
        # Liberar recursos de la cámara
        left_camera.stop()
        left_camera.release()
        right_camera.stop()
        right_camera.release()
        cv2.destroyAllWindows()
                

