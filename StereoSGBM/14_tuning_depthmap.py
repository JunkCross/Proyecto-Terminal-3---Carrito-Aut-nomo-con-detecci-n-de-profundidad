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

    blockSize = variable_mapping["SWS"]
    #blockSize es el SAD Window Size
    #Configuración del filtro
    sbm = cv2.StereoSGBM_create(
        minDisparity = variable_mapping['MinDisp'],
        numDisparities=64, 
        blockSize = blockSize,
        P1 = variable_mapping['P1'] * blockSize * blockSize,
        P2 = variable_mapping['P2'] * blockSize * blockSize,
        disp12MaxDiff = variable_mapping['disp_12_max_diff'],
        preFilterCap = variable_mapping['pre_filter_cap'],
        uniquenessRatio = variable_mapping['uniqueness_ratio'],
        speckleWindowSize = variable_mapping['speckle_windows_size'],
        speckleRange = variable_mapping['speckle_range'],
        mode = 0
    )
    
    

    #c, r = rectified_pair[0].shape
    dmLeft = cv2.resize(rectified_pair[0], resize)
    dmRight = cv2.resize(rectified_pair[1], resize)
    #dmLeft = rectified_pair[0]
    #dmRight = rectified_pair[1]
    disparity = sbm.compute(dmLeft, dmRight)


    #//Esto es agregado hasta aqui
    disparity = cv2.dilate(disparity, None, iterations=1)
    disparity = cv2.resize(disparity, (width, height))
    #//Esto aumenta el tamaño de la disparidad
    
    
    disparity_normalized = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
    #Convering Numpy Array to CV_8UC1
    image = np.array(disparity_normalized, dtype = np.uint8)
    disparity_color = cv2.applyColorMap(image, cv2.COLORMAP_JET)
    #h, w, _ = disparity_color.shape
    #print("TAmaño de la imagen de disparity_color: ", (h, w))
    
    return disparity_color, disparity_normalized


def save_load_map_settings(current_savalanjetsone, current_load, variable_mapping):
    global loading
    if current_save != 0:
        print('Guardando en el archivo...')
        
        result = json.dumps({'minDisparity':variable_mapping["MinDisp"], 
                             'SADWindowSize':variable_mapping["SWS"], 
                             'P1_filter':variable_mapping['P1'], 
                             'P2_filter':variable_mapping['P2'], 
                             'disp12MaxDiff': variable_mapping['disp_12_max_diff'], 
                             'preFilterCap':variable_mapping['pre_filter_cap'], 
                             'uniquenessRatio':variable_mapping['uniqueness_ratio'], 
                             'speckleWindowSize':variable_mapping['speckle_windows_size'],
                             'speckleRange':variable_mapping['speckle_range'], 
                             }, sort_keys=True, indent=4, separators=(',',':'))

        
        fName = '/home/alan/Documentos/StereoVision_HOME_CUDA/calibracion/3dmap_set_SGBM.txt'
        f = open (str(fName), 'w')
        f.write(result)
        f.close()
        print ('Configuración guardada en el archivo '+fName)


    if current_load != 0:
        if os.path.isfile('/home/alan/Documentos/StereoVision_HOME_CUDA/calibracion/3dmap_set_SGBM.txt') == True:
            loading = True
            fName = '/home/alan/Documentos/StereoVision_HOME_CUDA/calibracion/3dmap_set_SGBM.txt'
            print('Cargando parámetros desde el archivo...')
            f=open(fName, 'r')
            data = json.load(f)

            cv2.setTrackbarPos("MinDisp", "Stereo", data['minDisparity']+100)
            cv2.setTrackbarPos("SWS", "Stereo", data['SADWindowSize'])
            cv2.setTrackbarPos("P1", "Stereo", data['P1_filter'])
            cv2.setTrackbarPos("P2", "Stereo", data['P2_filter'])
            cv2.setTrackbarPos("disp_12_max_diff", "Stereo", data['disp12MaxDiff'])
            
            cv2.setTrackbarPos("pre_filter_cap", "Stereo", data['preFilterCap'])
            cv2.setTrackbarPos("uniqueness_ratio", "Stereo", data['uniquenessRatio'])
            cv2.setTrackbarPos("speckle_windows_size", "Stereo", data['speckleWindowSize'])
            cv2.setTrackbarPos("speckle_range", "Stereo", data['speckleRange'])

            f.close()
            print ('Parametros cargados desde el archivo '+fName)
            print ('Redibujar el mapa de profundidad con parámetros cargados...')
            print ('Hecho!') 
        else: 
            print ("El archivo para cargar no existe.")
            
            


def activateTrackbars(x):
    global loading
    loading = False


def create_trackbars() :
    global loading

    #SWS no puede ser mayor que el ancho y el alto de la imagen.
    #En este caso, ancho = 640 y alto = 480
    cv2.createTrackbar("MinDisp", "Stereo", -100, 200, activateTrackbars)
    cv2.createTrackbar("SWS", "Stereo", 3, 230, activateTrackbars)
    cv2.createTrackbar("P1", "Stereo", 8, 100, activateTrackbars)
    cv2.createTrackbar("P2", "Stereo", 32, 100, activateTrackbars)
    cv2.createTrackbar("disp_12_max_diff", "Stereo", -100, 100, activateTrackbars)
    cv2.createTrackbar("pre_filter_cap", "Stereo", 0, 63, activateTrackbars)
    cv2.createTrackbar("uniqueness_ratio", "Stereo", 0, 20, activateTrackbars)
    cv2.createTrackbar("speckle_windows_size", "Stereo", 0, 63, activateTrackbars)
    cv2.createTrackbar("speckle_range", "Stereo", 0, 40, activateTrackbars)
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
        # Se inicializa el trackbars y windows
        cv2.namedWindow("Stereo")
        create_trackbars()

        print ("Camaras iniciadas")

        variables = ["MinDisp", "SWS", "P1", "P2", "disp_12_max_diff", "pre_filter_cap", "uniqueness_ratio", 
                     "speckle_windows_size", "speckle_range"]

        
        variable_mapping = {"MinDisp": -25, "SWS" : 3, "P1" : 8, "P2" : 32, "disp_12_max_diff" : 1, "pre_filter_cap" : 30, 
                            "uniqueness_ratio" : 10, "speckle_windows_size" : 10, "speckle_range" : 15}

        

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
                        if v == "uniqueness_ratio" or v == "pre_filter_cap":
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
                if k == ord('q') or k == ord('Q'):
                    break

                else:
                    continue
    except Exception as e:
        print("Ocurrio un error:", str(e))


    finally:
        # Liberar recursos de la cámara
        left_camera.stop()
        left_camera.release()
        right_camera.stop()
        right_camera.release()
        cv2.destroyAllWindows()