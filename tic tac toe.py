#from mlf_api import RobotClient
#import time
#import cv2

#import numpy as np

#robot = RobotClient("spike.local")
#robot.connectWebRTC()
#robot.get_frame()
#robot.closeWebRTC()

#robot.move_xyz(cx+275,cy+300,0,[30,0,40])


def cambio_coord(cx,cy,cz):

        ypx = (305-cx)*1.55
        xpx = (434-cy)*1.4
        x = 50*xpx/65
        y = 50*ypx/65
        z = cz 
        offset = [65, 0, 75]
        q3 = 0
    
        #robot.move_xyz(x, y, z, offset, q3)

        return x,y,z,offset,q3


#Tablero
t_esquina_1=cambio_coord(150,50,20) #X,Y,Z
t_esquina_2=cambio_coord(360,50,20)
t_esquina_3=cambio_coord(150,260,20) #Izquierda a derecha y de arriba a abajo
t_esquina_4=cambio_coord(360,260,20)

for i in range (1,9):
    
    def casilla(cax,cay,caz): 
        
        i_esquina_1=cambio_coord(cax,cay,caz)
        i_esquina_2=cambio_coord(cax+70,cay,caz)
        i_esquina_3=cambio_coord(cax,cay+70,caz)
        i_esquina_4=cambio_coord(cax+70,cay+70,caz)

        return i_esquina_1,i_esquina_2,i_esquina_3,i_esquina_4


1_esquina_1, 1_esquina_2, 1_esquina_3, 1_esquina_4 = casilla(150,40,20)

1_esquina_1, 1_esquina_2, 1_esquina_3, 1_esquina_4 = casilla(150,40,20)
1_esquina_1, 1_esquina_2, 1_esquina_3, 1_esquina_4 = casilla(150,40,20)
1_esquina_1, 1_esquina_2, 1_esquina_3, 1_esquina_4 = casilla(150,40,20)
1_esquina_1, 1_esquina_2, 1_esquina_3, 1_esquina_4 = casilla(150,40,20)
1_esquina_1, 1_esquina_2, 1_esquina_3, 1_esquina_4 = casilla(150,40,20)
