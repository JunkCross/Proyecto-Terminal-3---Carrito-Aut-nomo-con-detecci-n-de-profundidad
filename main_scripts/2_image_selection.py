import cv2 
import os

# Variables globales preestablecidas
total_photos = 50
photo_height = 1232
photo_width = 1640
img_height = 480
img_width = 640


def SeperateImages():
    photo_counter = 1
    
    if (os.path.isdir("/home/jetson/Documents/StereoVision_HOME_CUDA/calibracion/pairs") == False):
        os.makedirs("/home/jetson/Documents/StereoVision_HOME_CUDA/calibracion/pairs")
        
    while photo_counter != total_photos:
        k = None
        filename = '/home/jetson/Documents/StereoVision_HOME_CUDA/calibracion/images/image_'+ str(photo_counter).zfill(2) + '.png'
        if os.path.isfile(filename) == False:
            print("Ningun archivo nombrado " + filename)
            photo_counter += 1
            
            continue
        pair_img = cv2.imread(filename, -1)
        
        print ("Image Pair: " + str(photo_counter))
        cv2.imshow("ImagePair", pair_img)
        
        #espera a que se presione cualquier tecla
        k = cv2.waitKey(0) & 0xFF
 
        if k == ord('y'):
            # guardar la foto
            imgLeft = pair_img[0:img_height, 0:img_width]  # Y+H and X+W
            imgRight = pair_img[0:img_height, img_width:photo_width]
            leftName = '/home/jetson/Documents/StereoVision_HOME_CUDA/calibracion/pairs/left_' + str(photo_counter).zfill(2) + '.png'
            rightName = '/home/jetson/Documents/StereoVision_HOME_CUDA/calibracion/pairs/right_' + str(photo_counter).zfill(2) + '.png'
            cv2.imwrite(leftName, imgLeft)
            cv2.imwrite(rightName, imgRight)
            print('Par No ' + str(photo_counter) + ' saved.')
            photo_counter += 1
     
        elif k == ord('n'):
            # salta la foto
            photo_counter += 1
            print ("Saltar")
            
        elif k == ord('q'):
            break  
  

            
    
    print('Fin de ciclo')
    
if __name__ == '__main__':

    print ("Se mostrarán las imágenes emparejadas")
    print ("Presione Y para aceptar y guardar la imagen")
    print ("Presione N para omitir la imagen si está borrosa/poco clara/cortada") 
    SeperateImages()



