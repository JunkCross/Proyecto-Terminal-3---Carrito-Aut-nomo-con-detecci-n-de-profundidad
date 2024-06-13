import time
import RPi.GPIO as GPIO
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


# Define las constantes del control PID (ajústalas según sea necesario)
KP = 0.1
KI = 0.01
KD = 0.01

# Configurar los pines de los sensores
sensores = [12, 13, 15, 16]

# Configurar los parámetros físicos de las ruedas y encoders
radio_rueda_cm = 3
segmentos_encoder = 20
circunferencia_rueda_cm = 2 * 3.141592 * radio_rueda_cm
distancia_por_segmento_cm = circunferencia_rueda_cm / segmentos_encoder

# Diccionarios para almacenar la información de cada sensor
interrupciones = {}
tiempo_anterior = {}
velocidad = {}
distancia_recorrida = {}
rpm = {}

# Configuración de GPIO
GPIO.setmode(GPIO.BOARD)
for sensor in sensores:
    GPIO.setup(sensor, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
    interrupciones[sensor] = 0
    tiempo_anterior[sensor] = time.time()
    velocidad[sensor] = 0
    distancia_recorrida[sensor] = 0
    rpm[sensor] = 0

# Función para manejar interrupciones
def manejar_interrupcion(pin):
    global interrupciones
    interrupciones[pin] += 1

# Asignar la función de manejo de interrupciones a cada pin de sensor
for sensor in sensores:
    GPIO.add_event_detect(sensor, GPIO.RISING, callback=manejar_interrupcion, bouncetime=20)



GPIO.setwarnings(False)
GPIO.setmode(GPIO.BOARD)


class Robot():
    def __init__(self, *args, **kwargs):
        super(Robot, self).__init__(*args, **kwargs)
        self.left_motor = [35, 36]
        self.right_motor = [37, 38]
        self.left_speed = 0
        self.right_speed = 0
        GPIO.setup(32, GPIO.OUT)
        GPIO.setup(33, GPIO.OUT) 
        self.pwm = [GPIO.PWM(32, 50), GPIO.PWM(33, 50)]
        GPIO.setup(self.left_motor[0], GPIO.OUT, initial=GPIO.LOW)
        GPIO.setup(self.right_motor[0], GPIO.OUT, initial=GPIO.LOW) 
        GPIO.setup(self.left_motor[1], GPIO.OUT, initial=GPIO.LOW)
        GPIO.setup(self.right_motor[1], GPIO.OUT, initial=GPIO.LOW) 
        self.pwm[0].start(0)
        self.pwm[1].start(0)
        self.left_last_error = 0
        self.right_last_error = 0
        self.left_integral = 0
        self.right_integral = 0

    def set_motors_pid(self, left_speed=-0.53, right_speed=-0.53, left_desired_speed=-0.54, right_desired_speed=-0.54):
        # Calcular la velocidad actual de las ruedas utilizando la información de los sensores
        left_actual_speed = self.calcular_velocidad_actual(sensores[0])  # Suponiendo que el primer sensor corresponde a la rueda izquierda
        right_actual_speed = self.calcular_velocidad_actual(sensores[2])  # Suponiendo que el tercer sensor corresponde a la rueda derecha
        #print('left: '+str(left_actual_speed), 'right: '+str(right_actual_speed))

        # Calcula el error entre la velocidad deseada y la velocidad medida
        left_error = left_desired_speed - left_actual_speed
        right_error = right_desired_speed - right_actual_speed
        
        # Calcula los términos proporcionales, integrales y derivativos
        left_p = KP * left_error
        right_p = KP * right_error
        left_i = KI * (left_error + self.left_integral)
        right_i = KI * (right_error + self.right_integral)
        left_d = KD * (left_error - self.left_last_error)
        right_d = KD * (right_error - self.right_last_error)
        
        # Calcula la velocidad ajustada utilizando el control PID
        left_adjusted_speed = left_speed + left_p + left_i + left_d
        right_adjusted_speed = right_speed + right_p + right_i + right_d
        
        # Limitar las velocidades ajustadas dentro del rango deseado
        left_adjusted_speed = max(min(left_adjusted_speed, 1.0), -0.72)
        right_adjusted_speed = max(min(right_adjusted_speed, 1.0), -0.72)
        
        # Aplica los ajustes de velocidad a los motores
        self.set_motors(left_adjusted_speed, right_adjusted_speed)
        
        # Actualiza los valores para la próxima iteración
        self.left_last_error = left_error
        self.right_last_error = right_error
        self.left_integral += left_error
        self.right_integral += right_error

    def calcular_velocidad_actual(self, sensor_pin):
        tiempo_actual = time.time()
        intervalo_tiempo = tiempo_actual - tiempo_anterior[sensor_pin]

        # Calcular velocidad
        velocidad_actual = (interrupciones[sensor_pin] * distancia_por_segmento_cm) / intervalo_tiempo

        # Actualizar tiempo anterior y reiniciar contador de interrupciones
        tiempo_anterior[sensor_pin] = tiempo_actual
        interrupciones[sensor_pin] = 0

        return velocidad_actual

    def set_motors(self, left_speed=1.0, right_speed=1.0):
        GPIO.output(self.left_motor[0], GPIO.HIGH)
        GPIO.output(self.right_motor[0], GPIO.HIGH) 
        self.left_speed = ((left_speed - (-1)) / 2) * 100
        self.right_speed = ((right_speed - (-1)) / 2) * 100
        self.pwm[0].ChangeDutyCycle(self.left_speed)
        self.pwm[1].ChangeDutyCycle(self.right_speed)
        
    def forward(self, speed=1.0, duration=None):
        GPIO.output(self.left_motor[0], GPIO.HIGH)
        GPIO.output(self.right_motor[0], GPIO.HIGH) 
        GPIO.output(self.left_motor[1], GPIO.LOW)
        GPIO.output(self.right_motor[1], GPIO.LOW) 
        self.speed = ((speed - (-1)) / 2) * 100
        self.pwm[0].ChangeDutyCycle(self.speed)
        self.pwm[1].ChangeDutyCycle(self.speed)

    def backward(self, speed=1.0):
        GPIO.output(self.left_motor[0], GPIO.LOW)
        GPIO.output(self.right_motor[0], GPIO.LOW) 
        GPIO.output(self.left_motor[1], GPIO.HIGH)
        GPIO.output(self.right_motor[1], GPIO.HIGH) 
        self.speed = ((speed - (-1)) / 2) * 100
        self.pwm[0].ChangeDutyCycle(self.speed)
        self.pwm[1].ChangeDutyCycle(self.speed)

    def left(self, speed=1.0):
        GPIO.output(self.left_motor[0], GPIO.LOW)
        GPIO.output(self.right_motor[0], GPIO.HIGH) 
        GPIO.output(self.left_motor[1], GPIO.HIGH)
        GPIO.output(self.right_motor[1], GPIO.LOW) 
        self.speed = ((speed - (-1)) / 2) * 100
        self.pwm[0].ChangeDutyCycle(self.speed)
        self.pwm[1].ChangeDutyCycle(self.speed)

    def right(self, speed=1.0):
        GPIO.output(self.left_motor[0], GPIO.HIGH)
        GPIO.output(self.right_motor[0], GPIO.LOW) 
        GPIO.output(self.left_motor[1], GPIO.LOW)
        GPIO.output(self.right_motor[1], GPIO.HIGH) 
        self.speed = ((speed - (-1)) / 2) * 100
        self.pwm[0].ChangeDutyCycle(self.speed)
        self.pwm[1].ChangeDutyCycle(self.speed)

    def stop(self):
        GPIO.output(self.left_motor[0], GPIO.LOW)
        GPIO.output(self.right_motor[0], GPIO.LOW) 
        GPIO.output(self.left_motor[1], GPIO.LOW)
        GPIO.output(self.right_motor[1], GPIO.LOW) 
        self.left_speed = 0
        self.right_speed = 0
        self.pwm[0].ChangeDutyCycle(self.left_speed)
        self.pwm[1].ChangeDutyCycle(self.right_speed)


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
    disparity_color = cv2.applyColorMap(image, cv2.COLORMAP_JET)
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

    # id 88 es el oso de peluche
    # id 79 cellphone
    # id 1 es persona
    if item_class == 88:  # Si el objeto detectado es un OSO DE FELPA
        
        print("Ose de felpa esta a: {}cm de distancia".format(distance))
        if distance > 45:  # Si la distancia es mayor que 30 cm, seguir al objeto
            
            carrito.set_motors(-0.2, -0.2)
            print("AVANZADO CARRITO")
        else:  # Si la distancia es igual o menor que 30 cm, detener el carrito
            carrito.stop()
            print("DETENIDO CARRITO")
        
    else:  # Si no se detecta ningún objeto (o el ID no es 88)
        # Detén el auto porque el osito de peluche no está a la vista.
        carrito.stop()
        print("OSO DE PELUCHE NO DETECTADO")
        print("CARRITO SE DETUVO POR FALTA DE OBJETO")




# Configuración del carrito
carrito = Robot()

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

        #cv2.namedWindow("DepthMap")

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
                object_detected = False
                if len(detections):
                    for item in detections:
                        if item.ClassID == 88:  # Si se detecta el objeto oso de peluche
                            objectDetection(item, disparity_normalized)
                            object_detected = True  # Establece el indicador en Verdadero
                            break
                        #else:
                        #    print("\nPARAR CARRITO")
                        #    carrito.stop()
                if not object_detected:
                    print("\nPARAR CARRITO")
                    carrito.stop()
                
     


                left_stacked = cv2.addWeighted(left_frame, 0.5, disparity_color, 0.5, 0.0)
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
        carrito.stop()
        GPIO.cleanup()