from mlf_api import RobotClient
import time
import cv2
import numpy as np

robot = RobotClient("192.168.0.106")

def show(frame):
    cv2.imshow("XD", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def show_contours(frame, contours):
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    show(frame)

def show(frame):
    cv2.imshow("XD", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def ocupar_casilla_jugador():

    robot.connectWebRTC() 
    
    try:
        frame = robot.get_frame()
        cv2.imwrite("frame.jpg", frame)
        show(frame)

        imagen = cv2.imread("frame.jpg")
        alto, ancho, canales = imagen.shape
        parte_alto = alto // 3
        parte_ancho = ancho // 3

        contador = 1
        for i in range(3):
            for j in range(3):
            # Calcular las coordenadas de la parte actual
                inicio_y = i * parte_alto
                fin_y = (i + 1) * parte_alto
                inicio_x = j * parte_ancho
                fin_x = (j + 1) * parte_ancho

        parte = imagen[inicio_y:fin_y, inicio_x:fin_x]

        nombre_archivo = f'parte{contador}.jpg'
        cv2.imwrite(nombre_archivo, parte)

        

        rgbImage = cv2.cvtColor(imagen, cv2.COLOR_RGB2BGR)

        lowerLimit = np.array([50, 20, 40])
        upperLimit = np.array([80, 50, 90])
        mask = cv2.inRange(rgbImage, lowerLimit, upperLimit)
        show(mask)

        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        print(f"N° de Contornos: {len(contours)}")

        show_contours(frame.copy(), contours)

        min_area = 1000
        contours = [c for c in contours if cv2.contourArea(c) < min_area]

        show_contours(frame, contours)

        c = max(contours, key=cv2.contourArea)
        M = cv2.moments(c)
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        print(f"Centroide: ({cx}, {cy})")

        cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)
        show(frame)

        #Condiciones para ocupar casilla
        if (cx>150 and cx<220):

            if (cy>50 and cy<120):
                return 1
            
            elif (cy>120 and cy<190):
                return 4

            elif (cy>190 and cy<270):
                return 7  
            
            else:
                return None 
            
        elif (cx>220 and cx<290):

            if (cy>50 and cy<120):
                return 2
            
            elif (cy>120 and cy<190):
                return 5

            elif (cy>190 and cy<270):
                return 8  
            
            else:
                return None
            
        elif (cx>290 and cx<360):

            if (cy>50 and cy<120):
                return 3
            
            elif (cy>120 and cy<190):
                return 6

            elif (cy>190 and cy<270):
                return 9  
            
            else:
                return None
            
        else:
            return None
        

    except Exception as e: 
        print(e)

    finally:
        robot.closeWebRTC()

def ocupar_casilla_robot():
    
    robot.connectWebRTC() 
    
    try:
        frame = robot.get_frame()
        cv2.imwrite("frame.jpg", frame)
        show(frame)

        rgbImage = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        lowerLimit = np.array([50, 20, 40])
        upperLimit = np.array([80, 50, 90])
        mask = cv2.inRange(rgbImage, lowerLimit, upperLimit)
        show(mask)

        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        print(f"N° de Contornos: {len(contours)}")

        show_contours(frame.copy(), contours)

        min_area = 1500
        contours = [c for c in contours if cv2.contourArea(c) > min_area]

        show_contours(frame, contours)

        c = max(contours, key=cv2.contourArea)
        M = cv2.moments(c)
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        print(f"Centroide: ({cx}, {cy})")

        cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)
        show(frame)

        #Condiciones para ocupar casilla
        if (cx>150 and cx<220):

            if (cy>50 and cy<120):
                return 1
            
            elif (cy>120 and cy<190):
                return 4

            elif (cy>190 and cy<270):
                return 7  
            
            else:
                return None 
            
        elif (cx>220 and cx<290):

            if (cy>50 and cy<120):
                return 2
            
            elif (cy>120 and cy<190):
                return 5

            elif (cy>190 and cy<270):
                return 8  
            
            else:
                return None
            
        elif (cx>290 and cx<360):

            if (cy>50 and cy<120):
                return 3
            
            elif (cy>120 and cy<190):
                return 6

            elif (cy>190 and cy<270):
                return 9  
            
            else:
                return None
            
        else:
            return None
        

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
        time.sleep(1)
        return x,y,z,offset,q3


#Marcar casillas (el robot marca una casilla)
def marca_casilla_1():
     
   
    cambio_coord_mov(290,190,20)
    cambio_coord_mov(185,85,-55)

    return None 

def marca_casilla_2():

     
    cambio_coord_mov(290,190,20)
    cambio_coord_mov(255,85,-55)

    return None 

def marca_casilla_3():

     
    cambio_coord_mov(290,190,20)
    cambio_coord_mov(325,85,-55)

    return None 

def marca_casilla_4():
    
    cambio_coord_mov(290,190,20)
    cambio_coord_mov(185,155,-40)
    return None 

def marca_casilla_5():
    
    cambio_coord_mov(290,190,20)
    cambio_coord_mov(255,155,-40)
    return None 

def marca_casilla_6():
    
    cambio_coord_mov(290,190,20)
    cambio_coord_mov(325,155,-40)
    return None 

def marca_casilla_7():
    
    cambio_coord_mov(290,190,20)
    cambio_coord_mov(185,225,-40)
    return None 

def marca_casilla_8():
    
    cambio_coord_mov(290,190,20)
    cambio_coord_mov(255,225,-40)
    return None 

def marca_casilla_9():
    
    cambio_coord_mov(290,190,20)
    cambio_coord_mov(325,225,-40)
    return None 



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


#marca_casilla_1()
#marca_casilla_2()
#marca_casilla_3()
#marca_casilla_4()
#marca_casilla_5()
#marca_casilla_6()
#marca_casilla_7()
#marca_casilla_8()
#marca_casilla_9()


#time.sleep(15)
#ocupar_casilla_jugador()
#time.sleep(5)
#ocupar_casilla_robot()

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









def casilla_marc(cax,cay,caz): 
        
    num_esquina_1=cambio_coord_mov(cax,cay,caz)
    num_esquina_2=cambio_coord_mov(cax+70,cay,caz)
    num_esquina_3=cambio_coord_mov(cax+70,cay+70,caz) # Forma una casilla
    num_esquina_4=cambio_coord_mov(cax,cay+70,caz)
    num_esquina_5=cambio_coord_mov(cax,cay,caz)

    return None




