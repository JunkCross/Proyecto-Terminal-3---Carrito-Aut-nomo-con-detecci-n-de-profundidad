import cv2
import numpy as np
import json
from stereovision.calibration import StereoCalibration
from start_cameras import Start_Cameras


# Preajuste predeterminado del mapa de profundidad
SWS = 5
PFS = 5
PFC = 29
MDS = -30
NOD = 160
TTH = 100
UR = 10
SR = 14
SPWS = 100


def load_map_settings(file):
    global SWS, PFS, PFC, MDS, NOD, TTH, UR, SR, SPWS, loading_settings, sbm
    print('Cargando parámetros desde el archivo...')
    f = open(file, 'r')
    data = json.load(f)
    #cargando datos del archivo json y asignándolos a las Variables
    SWS = data['SADWindowSize']
    PFS = data['preFilterSize']
    PFC = data['preFilterCap']
    MDS = data['minDisparity']
    NOD = data['numberOfDisparities']
    TTH = data['textureThreshold']
    UR = data['uniquenessRatio']
    SR = data['speckleRange']
    SPWS = data['speckleWindowSize']
    
    #cambiar los valores reales de las variables
    sbm = cv2.StereoBM_create(numDisparities=64, blockSize=SWS) 
    sbm.setPreFilterType(1)
    sbm.setPreFilterSize(PFS)
    sbm.setPreFilterCap(PFC)
    sbm.setMinDisparity(MDS)
    sbm.setNumDisparities(NOD)
    sbm.setTextureThreshold(TTH)
    sbm.setUniquenessRatio(UR)
    sbm.setSpeckleRange(SR)
    sbm.setSpeckleWindowSize(SPWS)
    f.close()
    print('Parámetros cargados desde el archivo ' + file)

def stereo_depth_map(rectified_pair):
    #blockSize es el SAD Window Size
    
    #Prueba mitad de pixeles
    width = 640
    height = 480
    downsample = 2
    resize = (int(width/downsample), int(height/downsample)) 

    #//Esto reduce el tamaño dde los pares de la imagen
    dmLeft = cv2.resize(rectified_pair[0], resize)
    dmRight = cv2.resize(rectified_pair[1], resize)
    
    #print("Dimensiones de dmLeft:", dmLeft.shape)
    #print("Dimensiones de dmRight:", dmRight.shape)
    #dmLeft = rectified_pair[0]
    #dmRight = rectified_pair[1]
    disparity = sbm.compute(dmLeft, dmRight)
    
    
    #//Esto aumenta el tamaño de la disparidad
    disparity = cv2.dilate(disparity, None, iterations=1)
    disparity = cv2.resize(disparity, (width, height))
    #//Esto aumenta el tamaño de la disparidad
    
    
    disparity_normalized = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
    image = np.array(disparity_normalized, dtype = np.uint8)
    disparity_color = cv2.applyColorMap(image, cv2.COLORMAP_JET)
    return disparity_color, disparity_normalized

def onMouse(event, x, y, flag, disparity_normalized):
    if event == cv2.EVENT_LBUTTONDOWN:
        distance = disparity_normalized[y][x]
        print("Distancia en centimetros {}".format(distance))
        return distance


if __name__ == "__main__":
    left_camera = Start_Cameras(1).start()
    right_camera = Start_Cameras(0).start()
    load_map_settings('/home/alan/Documentos/StereoVision_HOME_CUDA/calibracion/3dmap_set.txt')

    try:
        
        cv2.namedWindow("DepthMap")

        while True:
            left_grabbed, left_frame = left_camera.read()
            right_grabbed, right_frame = right_camera.read()

            if left_grabbed and right_grabbed:  
                #Convertir BGR a escala de grises
                left_gray_frame = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
                right_gray_frame = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)

                #llamar a todos los resultados de calibración
                calibration = StereoCalibration(input_folder='/home/alan/Documentos/StereoVision_HOME_CUDA/calibracion/calib_result')
                rectified_pair = calibration.rectify((left_gray_frame, right_gray_frame))
                disparity_color, disparity_normalized = stereo_depth_map(rectified_pair)

                # Función de clic del mouse
                cv2.setMouseCallback("DepthMap", onMouse, disparity_normalized)
                
                #Prueba mitad de pixeles
                width = 640
                height = 480
                downsample = 2
                resize = (int(width/downsample), int(height/downsample))
                LeftDown = cv2.resize(left_frame, resize)
                
                
                output = cv2.addWeighted(left_frame, 0.5, disparity_color, 0.5, 0.0)
                #output = cv2.addWeighted(left_frame, 0.5, disparity_color, 0.5, 0.0)
                cv2.imshow("DepthMap", np.hstack((disparity_color, output)))
                cv2.imshow("Frame", np.hstack((rectified_pair[0], rectified_pair[1])))
                cv2.imshow("Frames", np.hstack((left_frame, right_frame)))

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