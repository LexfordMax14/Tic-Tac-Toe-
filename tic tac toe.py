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

#---Definimos el estado de las casilla----
def divide_image(image, rows, cols):
    height, width = image.shape[:2]
    row_height = height // rows
    col_width = width // cols
    cells = []
    for i in range(rows):
        for j in range(cols):
            x1, y1 = j * col_width, i * row_height
            x2, y2 = x1 + col_width, y1 + row_height
            cell = image[y1:y2, x1:x2]
            cells.append(cell)
    return cells

def is_cell_occupied(cell, lower_color, upper_color):
    mask = cv2.inRange(cell, lower_color, upper_color)
    return cv2.countNonZero(mask) > 0
   
def estatus_tablero():
    try:
    # Captura la imagen del tablero del robot
        frame = robot.get_frame()
        cv2.imwrite("frame.jpg", frame)
        show(frame)

        parte_tablero= "frame.jpg"[50:260, 120:360]   
        # Convierte la imagen a BGR
        rgbImage = cv2.cvtColor(parte_tablero, cv2.COLOR_RGB2BGR)

        # Divide la imagen del tablero en 9 casillas
        cells = divide_image(rgbImage, 3, 3)

        # Define los límites de color para detectar las marcas
        lower_color_x = np.array([50, 20, 40])  # Ejemplo de color para "X"
        upper_color_x = np.array([80, 50, 90])
        lower_color_o = np.array([30, 60, 20])  # Ejemplo de color para "O"
        upper_color_o = np.array([70, 120, 70])

        # Procesa cada casilla para determinar si está ocupada
        board_status = []
        for idx, cell in enumerate(cells):
            occupied_x = is_cell_occupied(cell, lower_color_x, upper_color_x)
            occupied_o = is_cell_occupied(cell, lower_color_o, upper_color_o)
            if occupied_x:
                board_status.append("X")
            elif occupied_o:
                board_status.append("O")
            else:
                board_status.append(" ")

        # Muestra el estado del tablero
        for i in range(3):
            print(board_status[i * 3:(i + 1) * 3])

    except Exception as e:
        print(f"Error: {e}")

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

    








