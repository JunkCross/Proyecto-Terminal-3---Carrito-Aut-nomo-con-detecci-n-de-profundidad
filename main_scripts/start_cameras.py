import cv2
import numpy as np
import threading

class Start_Cameras:

    def __init__(self, sensor_id):
        # Inicializar variables de instancia
        # Elemento de captura de video OpenCV
        self.video_capture = None
        # La última imagen capturada de la cámara.
        self.frame = None
        self.grabbed = False
        # El hilo donde corre la captura del video.
        self.read_thread = None
        self.read_lock = threading.Lock()
        self.running = False

        self.sensor_id = sensor_id

        gstreamer_pipeline_string = self.gstreamer_pipeline()
        self.open(gstreamer_pipeline_string)

    #Abriendo las cámaras
    def open(self, gstreamer_pipeline_string):
        gstreamer_pipeline_string = self.gstreamer_pipeline()
        try:
            self.video_capture = cv2.VideoCapture(
                gstreamer_pipeline_string, cv2.CAP_GSTREAMER
            )
            grabbed, frame = self.video_capture.read()
            print("Se abren cámaras")

        except RuntimeError:
            self.video_capture = None
            print("No se puede abrir la camara")
            print("Pipeline: " + gstreamer_pipeline_string)
            return
        # Tome el primer cuadro para comenzar la captura de video.
        self.grabbed, self.frame = self.video_capture.read()

    #Encendiendo las cámaras
    def start(self):
        if self.running:
            print('La captura de vídeo ya se está ejecutando.')
            return None
        # crear un hilo para leer la imagen de la cámara
        if self.video_capture != None:
            self.running = True
            self.read_thread = threading.Thread(target=self.updateCamera, daemon=True)
            self.read_thread.start()
        return self

    def stop(self):
        self.running = False
        self.read_thread.join()

    def updateCamera(self):
        # Este es el hilo para leer imágenes de la cámara.
        while self.running:
            try:
                grabbed, frame = self.video_capture.read()
                with self.read_lock:
                    self.grabbed = grabbed
                    self.frame = frame
            except RuntimeError:
                print("No se pudo leer la imagen de la cámara")

    def read(self):
        with self.read_lock:
            frame = self.frame.copy()
            grabbed = self.grabbed
        return grabbed, frame

    def release(self):
        if self.video_capture != None:
            self.video_capture.release()
            self.video_capture = None
        # Ahora mata el hilo
        if self.read_thread != None:
            self.read_thread.join()

    # Actualmente hay configuraciones de velocidad de fotogramas en la cámara CSI en Nano a través de gstreamer
    # Aquí seleccionamos directamente sensor_mode 3
    def gstreamer_pipeline(self,
            sensor_mode=3,
            capture_width=1640,
            capture_height=1232,
            display_width=640,
            display_height=480,
            framerate=30,
            flip_method=2,
    ):
        return (
                "nvarguscamerasrc sensor-id=%d sensor-mode=%d ! "
                "video/x-raw(memory:NVMM), "
                "width=(int)%d, height=(int)%d, "
                "format=(string)NV12, framerate=(fraction)%d/1 ! "
                "nvvidconv flip-method=%d ! "
                "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
                "videoconvert ! "
                "video/x-raw, format=(string)BGR ! appsink"
                % (
                    self.sensor_id,
                    sensor_mode,
                    capture_width,
                    capture_height,
                    framerate,
                    flip_method,
                    display_width,
                    display_height,
                )
        )


#Este es el principal.
if __name__ == "__main__":
    left_camera = Start_Cameras(0).start()
    right_camera = Start_Cameras(1).start()

    while True:
        left_grabbed, left_frame = left_camera.read()
        right_grabbed, right_frame = right_camera.read()

        if left_grabbed and right_grabbed:
            images = np.hstack((left_frame, right_frame))
            cv2.imshow("Camera Images", images)
            k = cv2.waitKey(1) & 0xFF

            if k == ord('q'):
                break
        else:
            break

    left_camera.stop()
    left_camera.release()
    right_camera.stop()
    right_camera.release()
    cv2.destroyAllWindows()