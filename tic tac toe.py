from mlf_api import RobotClient
import time
import cv2
import numpy as np

robot = RobotClient("192.168.0.106")

robot.connectWebRTC()


def show(frame):
    cv2.imshow("robot", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

try:
    frame = robot.get_frame()
    cv2.imwrite("frame.jpg", frame)
    show(frame)    
except Exception as e: 
    print(e)

finally:
    robot.closeWebRTC()



#------Cambio de coords-----
def cambio_coord(cx,cy,cz):

        ypx = (305-cx)*1.55
        xpx = (434-cy)*1.4
        x = 50*xpx/65
        y = 50*ypx/65
        z = cz 
        offset = [65, 0, 75]
        q3 = 0
    
        return x,y,z,offset,q3

def cambio_coord_mov(cx,cy,cz):

        ypx = (305-(cx))
        xpx = (434-(cy))
        x = 50*xpx/65
        y = 50*ypx/65
        z = cz 
        offset = [65, 0, 75]
        q3 = 0
        print(x,y,z)
        robot.move_xyz(x, y, z, offset, q3)
        time.sleep(5)
        return x,y,z,offset,q3

#-----dibuja tablero------ #X,Y,Z
cambio_coord_mov(150,50,20)  #Esquina Arriba a la izquierda
cambio_coord_mov(360,50,20)  
cambio_coord_mov(360,260,20)
cambio_coord_mov(150,260,20) #Izquierda arriba, derecha arriba, derecha abajo, izquierda abajo y cierra el cuadrado
cambio_coord_mov(150,50,20) #Forma un cuadrado grande


#-----dibuja casillas-----
cambio_coord_mov(150,50,30)  #levanta lapiz
cambio_coord_mov(220,50,20)  #se posiciona para marcar la primera linea
cambio_coord_mov(220,260,20) #primera linea lista

cambio_coord_mov(220,260,30) #levanta lapiz
cambio_coord_mov(290,260,20) #se posiciona para marcar la segunda linea
cambio_coord_mov(290,50,20) #segunda linea lista

cambio_coord_mov(290,50,30)  #levanta lapiz
cambio_coord_mov(360,120,20) #se posiciona para marcar la tercera linea
cambio_coord_mov(150,120,20) #tercera linea lista

cambio_coord_mov(150,120,30) #levanta lapiz
cambio_coord_mov(150,190,20) #se posiciona para marcar la cuarta linea
cambio_coord_mov(360,190,20) #cuarta linea lista



