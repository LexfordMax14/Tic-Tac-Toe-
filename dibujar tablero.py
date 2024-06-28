from mlf_api import RobotClient
import time
import cv2
import numpy as np

robot = RobotClient("192.168.0.106")


#-----Proceso robot------

#-----dibuja tablero------ #X,Y,Z
#cambio_coord_mov(0,0,0)
#cambio_coord_mov(150,50,-10)  #levantar lapiz
#cambio_coord_mov(150,50,-40)  #Esquina Arriba a la izquierda
#cambio_coord_mov(150,50,-10)  #levantar lapiz
#cambio_coord_mov(360,50,-10)  
#cambio_coord_mov(360,50,-40)
#cambio_coord_mov(360,50,-10)  
#cambio_coord_mov(360,260,-10)
#cambio_coord_mov(360,260,-45)
#cambio_coord_mov(360,260,-10)
#cambio_coord_mov(150,260,-10)
#cambio_coord_mov(150,260,-40) #Izquierda arriba, derecha arriba, derecha abajo, izquierda abajo y cierra el cuadrado
#cambio_coord_mov(150,260,-10)
#-----dibuja casillas-----

#cambio_coord_mov(220,50,-10)
#cambio_coord_mov(220,50,-40)#se posiciona para marcar la primera linea
#cambio_coord_mov(220,50,-10)
#cambio_coord_mov(290,50,-10)
#cambio_coord_mov(290,50,-40) #primera linea lista
#cambio_coord_mov(290,50,-10)#

#cambio_coord_mov(220,120,-10)
#cambio_coord_mov(220,120,-40)  #se posiciona para marcar la primera linea
#cambio_coord_mov(220,120,-10)
#cambio_coord_mov(290,120,-10)
#cambio_coord_mov(290,120,-40)
#cambio_coord_mov(220,50,-10)

#cambio_coord_mov(220,190,-10)
#cambio_coord_mov(220,190,-40)  #se posiciona para marcar la primera linea
#cambio_coord_mov(220,190,-10)
#cambio_coord_mov(290,190,-10)
#cambio_coord_mov(290,190,-45)
#cambio_coord_mov(290,190,-10)

#cambio_coord_mov(220,260,-10)
#cambio_coord_mov(220,260,-40)  #se posiciona para marcar la primera linea
#cambio_coord_mov(220,260,-10)
#cambio_coord_mov(290,260,-10)
#cambio_coord_mov(290,260,-40)
#cambio_coord_mov(290,260,-10)

#cambio_coord_mov(150,120,-10)
#cambio_coord_mov(150,120,-40)  #se posiciona para marcar la primera linea
#cambio_coord_mov(150,120,-10)
#cambio_coord_mov(360,120,-10)
#cambio_coord_mov(360,120,-40)
#cambio_coord_mov(360,120,-10)

#cambio_coord_mov(150,190,-10)
#cambio_coord_mov(150,190,-40)  #se posiciona para marcar la primera linea
#cambio_coord_mov(150,190,-10)
#cambio_coord_mov(360,190,-10)
#cambio_coord_mov(360,190,-40)
#cambio_coord_mov(360,190,-10)
