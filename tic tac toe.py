from mlf_api import RobotClient
import time
import cv2

import numpy as np

robot = RobotClient("spike.local")
robot.connectWebRTC()
robot.get_frame()
robot.closeWebRTC()

robot.move_xyz(cx+275,cy+300,0,[30,0,40])

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

        ypx = (305-cx)*1.55
        xpx = (434-cy)*1.4
        x = 50*xpx/65
        y = 50*ypx/65
        z = cz 
        offset = [65, 0, 75]
        q3 = 0
    
        robot.move_xyz(x, y, z, offset, q3)

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
cambio_coord_mov(360,50,20) #cuarta linea lista

cambio_coord_mov(0,0,60) #vuelve al 0,0









def casilla_marc(cax,cay,caz): 
        
    num_esquina_1=cambio_coord_mov(cax,cay,caz)
    num_esquina_2=cambio_coord_mov(cax+70,cay,caz)
    num_esquina_3=cambio_coord_mov(cax+70,cay+70,caz) # Forma una casilla
    num_esquina_4=cambio_coord_mov(cax,cay+70,caz)
    num_esquina_5=cambio_coord_mov(cax,cay,caz)

    return None



