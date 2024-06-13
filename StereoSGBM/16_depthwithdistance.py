import cv2
import numpy as np
import json
from stereovision.calibration import StereoCalibration
from start_cameras import Start_Cameras
import jetson.inference
import jetson.utils

# Preajuste predeterminado del mapa de profundidad
minDisparity = -30
SWS = 5
P1 = 5
P2 = 29
disp12MaxDiff =45
preFilterCap = 160
uniquenessRatio = 100
speckleWindowSize = 10
speckleRange = 14

#Distancia preestablecida
distance = 0

def load_map_settings(file):
    global MD, SWS, P1A, P2B, D12MD, PFC, UR, SWSWW, SR, loading_settings, sbm
    print('Cargando parámetros desde el archivo...')
    f = open(file, 'r')
    data = json.load(f)
    
    #cargando datos del archivo json y asignándolos a las Variables
    MD = data['minDisparity']
    SWS = data['SADWindowSize']
    P1A = data['P1_filter']
    P2B = data['P2_filter']
    D12MD = data['disp12MaxDiff']
    PFC = data['preFilterCap']
    UR = data['uniquenessRatio']
    SWSWW = data['speckleWindowSize']
    SR = data['speckleRange']
    
    
    #cambiar los valores reales de las variables
    sbm = cv2.StereoSGBM_create(
        minDisparity = MD, 
        numDisparities = 64, 
        blockSize = SWS,
        P1 = P1A * SWS * SWS,
        P2 = P2B * SWS * SWS,
        disp12MaxDiff = D12MD,
        preFilterCap = PFC,
        uniquenessRatio = UR,
        speckleWindowSize = SWSWW,
        speckleRange = SR
    )
    
    
    f.close()
    print('Parámetros cargados desde el archivo ' + file)

def stereo_depth_map(rectified_pair):
    #blockSize es el SAD Window Size
    
    #Prueba mitad de pixeles
    width = 640
    height = 480
    downsample = 2
    resize = (int(width/downsample), int(height/downsample)) 

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
    image = np.array(disparity_normalized, dtype = np.uint8)
    disparity_color = cv2.applyColorMap(image, cv2.COLORMAP_TURBO)
    return disparity_color, disparity_normalized



#Determine la distancia al píxel haciendo clic con el mouse
def onMouse(event, x, y, flag, disparity_normalized):
    if event == cv2.EVENT_LBUTTONDOWN:
        distance = disparity_normalized[y][x]
        print("Distancia en centimetros {}".format(distance))
        return distance

#Distancia a la persona mediante detección de objetos
def objectDetection(item, disparity_normalized):
    item_class = item.ClassID
    item_coords = item.Center
    x_coord = int(item_coords[0])
    y_coord = int(item_coords[1])
    distance = disparity_normalized[y_coord][x_coord]

    #para evitar la detección de diferentes objetos, solo nos centramos en personas que tienen un ClassID de 1
    if item_class == 1:
        print("Persona esta a: {}cm de distancia".format(distance))

# Modelo de detección de objetos
net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.5)

#net = jetson.inference.detectNet(argv=["--model=/home/aryan/StereoVision/SSD-Mobilenet-v2/ssd_mobilenet_v2_coco.uff",
#"--labels=/home/aryan/StereoVision/SSD-Mobilenet-v2/ssd_coco_labels.txt", 
#"--input-blob=Input", "--output-cvg=NMS", "--output-bbox=NMS_1"], threshold=0.5)



if __name__ == "__main__":
    left_camera = Start_Cameras(1).start()
    right_camera = Start_Cameras(0).start()
    load_map_settings("/home/alan/Documentos/StereoVision_HOME_CUDA/calibracion/3dmap_set_SGBM.txt")
    
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
            
            
                # Detección de objetos y distancia.
                left_cuda_frame = jetson.utils.cudaFromNumpy(left_frame)
                detections = net.Detect(left_cuda_frame)
                if len(detections):
                    for item in detections:
                        objectDetection(item, disparity_normalized )


                
                
                left_stacked = cv2.addWeighted(left_frame, 0.5, disparity_color, 0.5, 0.0)
                #left_stacked = cv2.addWeighted(left_frame, 0.5, disparity_color, 0.5, 0.0)
                cv2.imshow("DepthMap", np.hstack((disparity_color, left_stacked)))


                k = cv2.waitKey(1) & 0xFF
                if k == ord('q'):
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
                
