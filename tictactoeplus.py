from mlf_api import RobotClient
import time
import cv2
import numpy as np

# Conexión con el robot
robot = RobotClient("192.168.0.100")

# Función para mostrar una imagen
def show(frame):
    cv2.imshow("XD", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Función para mostrar los contornos detectados en la imagen
def show_contours(frame, contours):
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    show(frame)

# Función para dividir la imagen en una cuadrícula de celdas
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

# Función para verificar si una celda está ocupada basado en un rango de color
def is_cell_occupied(cell, lower_color, upper_color):
    mask = cv2.inRange(cell, lower_color, upper_color)
    return cv2.countNonZero(mask) > 0

# Función para capturar y procesar el estado del tablero
def estatus_tablero():
    try:
        # Captura la imagen del tablero del robot
        frame = robot.get_frame()
        cv2.imwrite("frame.jpg", frame)
        show(frame)

        # Convierte la imagen a escala de grises
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Aplica detección de bordes utilizando Canny
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)

        # Aplica transformada de Hough para detectar líneas rectas
        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)

        # Dibuja las líneas detectadas sobre la imagen original
        for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # Muestra las líneas detectadas en la imagen original
        show(frame)

        # Calcula los puntos extremos de las líneas para delimitar el área del tablero
        x_coords = []
        y_coords = []
        for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            x_coords.extend([x1, x2])
            y_coords.extend([y1, y2])

        min_x = min(x_coords)
        max_x = max(x_coords)
        min_y = min(y_coords)
        max_y = max(y_coords)

        # Recorta la imagen del área del tablero
        cropped_image = frame[min_y:max_y, min_x:max_x]
        show(cropped_image)

        # Guarda la imagen recortada
        cv2.imwrite("cropped_image.jpg", cropped_image)

        # Convierte la imagen a BGR
        rgbImage = cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR)

        # Divide la imagen del tablero en 9 celdas
        cells = divide_image(rgbImage, 3, 3)

        # Define los límites de color para detectar las marcas
        lower_color_x = np.array([56, 31, 0])  # Ejemplo de color para "X"
        upper_color_x = np.array([122, 141, 83])
        lower_color_o = np.array([0, 0, 111])  # Ejemplo de color para "O"
        upper_color_o = np.array([61, 128, 145])

        # Procesa cada celda para determinar si está ocupada
        board_status = []
        for idx, cell in enumerate(cells):
            occupied_x = is_cell_occupied(cell, lower_color_x, upper_color_x)
            occupied_o = is_cell_occupied(cell, lower_color_o, upper_color_o)
            if occupied_x:
                board_status.append(0)  # Usuario
            elif occupied_o:
                board_status.append(1)  # NPC
            else:
                board_status.append(None)

        # Muestra el estado del tablero
        for i in range(3):
            print(board_status[i * 3:(i + 1) * 3])

        return board_status

    except Exception as e:
        print(f"Error: {e}")

    finally:
        robot.closeWebRTC()

# Función para convertir coordenadas

def cambio_coord(cx, cy, cz):
    ypx = (305 - cx) * 1.55
    xpx = (434 - cy) * 1.55
    x = 50 * xpx / 65
    y = 50 * ypx / 65
    z = cz
    offset = [65, 0, 75]
    q3 = 0
    return x, y, z, offset, q3

# Función para mover el robot a una celda específica y marcar
def cambio_coord_mov(cx, cy, cz):
    ypx = (200 - cx) * 0.9
    xpx = (480 - cy) * 0.9
    x = 50 * xpx / 65
    y = 50 * ypx / 65
    z = cz 
    offset = [65, 0, 75]
    q3 = 0

    print(x, y, z)
    robot.move_xyz(x, y, z, offset, q3)
    robot.move_xyz(0, 250, 30, offset, q3)
    time.sleep(2)
    return x, y, z, offset, q3

# Funciones para marcar las casillas específicas
def marca_casilla_1():
    cambio_coord_mov(290, 190, 20)
    cambio_coord_mov(185, 85, -55)
    return None

def marca_casilla_2():
    cambio_coord_mov(290, 190, 20)
    cambio_coord_mov(255, 85, -55)
    return None

def marca_casilla_3():
    cambio_coord_mov(290, 190, 20)
    cambio_coord_mov(325, 85, -55)
    return None

def marca_casilla_4():
    cambio_coord_mov(290, 190, 20)
    cambio_coord_mov(185, 155, -40)
    return None

def marca_casilla_5():
    cambio_coord_mov(290, 190, 20)
    cambio_coord_mov(255, 155, -40)
    return None

def marca_casilla_6():
    cambio_coord_mov(290, 190, 20)
    cambio_coord_mov(325, 155, -40)
    return None

def marca_casilla_7():
    cambio_coord_mov(290, 190, 20)
    cambio_coord_mov(185, 225, -40)
    return None

def marca_casilla_8():
    cambio_coord_mov(290, 190, 20)
    cambio_coord_mov(255, 225, -40)
    return None

def marca_casilla_9():
    cambio_coord_mov(290, 190, 20)
    cambio_coord_mov(325, 225, -40)
    return None

# --- Código del juego de Tic-Tac-Toe ---

# Importaciones necesarias para el juego de Tic-Tac-Toe
from random import randint, choice

winner_lines = [[0, 1, 2], [3, 4, 5], [6, 7, 8],
                [0, 3, 6], [1, 4, 7], [2, 5, 8],
                [0, 4, 8], [2, 4, 6]]

# Clase para la política de decisiones aleatoria
class Random:
    def __init__(self, state):
        self.state = state

    @property
    def best_action(self):
        return choice([i for i, x in enumerate(self.state) if x is None])

# Clase del juego de Tic-Tac-Toe
class TicTacToe:
    def __init__(self, npc_policy=Random, board=None):
        self.board = board if board is not None else [None] * 9
        self.turn = randint(0, 1)
        self.npc_policy = npc_policy

    @property
    def finished(self):
        if self.winner_user:
            return True, 'Ganaste!'
        elif self.winner_npc:
            return True, 'Te ganó el computador!'
        elif self.board.count(None) == 0:
            return True, 'Empate'
        return False, ''

    @property
    def winner_user(self):
        return winner(self.board, 0)

    @property
    def winner_npc(self):
        return winner(self.board, 1)

    def run(self):
        print('Tu: X | PC: O')
        print(self, '\n')
        while not self.finished[0]:
            if self.turn == 0:
                print('Tu turno')
                self.play_user()
            else:
                print('Turno del PC')
                self.play_npc()
            self.turn = 1 - self.turn
            print('', self, '', sep='\n')
        print(self.finished[1])

    def play_user(self):
        index = bounded_numeric_input('Indica tu jugada (1-9): ', 1, 9) - 1
        while self.board[index] is not None:
            index = bounded_numeric_input('Indica tu jugada (1-9): ', 1, 9) - 1
        self.board[index] = 0

    def play_npc(self):
        policy = self.npc_policy(self.board)
        index = policy.best_action
        self.board[index] = 1
        self.mark_board(index)

    def mark_board(self, index):
        if index == 0:
            marca_casilla_1()
        elif index == 1:
            marca_casilla_2()
        elif index == 2:
            marca_casilla_3()
        elif index == 3:
            marca_casilla_4()
        elif index == 4:
            marca_casilla_5()
        elif index == 5:
            marca_casilla_6()
        elif index == 6:
            marca_casilla_7()
        elif index == 7:
            marca_casilla_8()
        elif index == 8:
            marca_casilla_9()

    def __str__(self):
        _ = ['X', 'O', '_']
        board = self.board if self.board.count(None) != 9 else [2] * 9
        string = ''
        for i in range(2, 9, 3):
            row = [_[i] if i is not None else ' ' for i in board[i - 2:i + 1]]
            string += '|'.join(row) + f'   -   {i-1}|{i}|{i+1}\n'
        return string[:-1]

# Función para verificar si hay un ganador
def winner(state, player):
    return any(all(state[c] == player for c in line) for line in winner_lines)

# Función para obtener un valor numérico dentro de un rango específico
def bounded_numeric_input(prompt, lower, upper):
    value = int(input(prompt))
    while value < lower or value > upper:
        print(f'Por favor ingresa un número entre {lower} y {upper}')
        value = int(input(prompt))
    return value

# Clase para el árbol de decisión MiniMax
class Node:
    def __init__(self, value, parent=None):
        self.value = value
        self.parent = parent
        self.leaf = False
        self.score = None
        self.children = []

    def add_child(self, node):
        self.children.append(node)

from tictactoe import TicTacToe, winner
from time_decorator import time
inf = float('infinity')

@time('Tiempo de decisión:')
class MiniMaxTree:
    def __init__(self, root):
        self.root = root if isinstance(root, Node) else Node(root)
        self.build(self.root, 1, -inf, +inf)

    def build(self, node, turn, alpha, beta):
        states = next_states(node.value, turn)
        if not states:
            if winner(node.value, 1):
                node.score = 1
            elif winner(node.value, 0):
                node.score = -1
            else:
                node.score = 0
            return node.score
        if turn == 1:
            for state in states:
                child = Node(state)
                node.add_child(child)
                alpha = max(alpha, self.build(child, 0, alpha, beta))
                if beta <= alpha:
                    break
            score = alpha
        else:
            for state in states:
                child = Node(state)
                node.add_child(child)
                beta = min(beta, self.build(child, 1, alpha, beta))
                if beta <= alpha:
                    break
            score = beta
        node.score = score
        return node.score

    @property
    def best_action(self):
        for child in self.root.children:
            if child.score == self.root.score:
                choice = child
                break
        state = self.root.value
        next_state = choice.value
        for i in range(9):
            if state[i] != next_state[i]:
                return i
        raise Exception('States are equal or different in size')

# Función para verificar si el estado es terminal
def terminal(state):
    return winner(state, 1) or winner(state, 0) or state.count(None) == 0

# Función para generar los siguientes estados posibles
def next_states(state, turn):
    indices = [i for i, x in enumerate(state) if x is None]
    if not terminal(state):
        return [state[:i] + [turn] + state[i + 1:] for i in indices]
    return []

# Ejecución del juego
if __name__ == '__main__':
    initial_board = estatus_tablero()
    g = TicTacToe(npc_policy=MiniMaxTree, board=initial_board)
    g.run()


#-----Proceso robot------

#-----dibuja tablero------ #X,Y,Z
cambio_coord_mov(0,0,0)
cambio_coord_mov(150,50,-10)  #levantar lapiz
cambio_coord_mov(150,50,-40)  #Esquina Arriba a la izquierda
cambio_coord_mov(150,50,-10)  #levantar lapiz
cambio_coord_mov(360,50,-10)  
cambio_coord_mov(360,50,-40)
cambio_coord_mov(360,50,-10)  
cambio_coord_mov(360,260,-10)
cambio_coord_mov(360,260,-45)
cambio_coord_mov(360,260,-10)
cambio_coord_mov(150,260,-10)
cambio_coord_mov(150,260,-40) #Izquierda arriba, derecha arriba, derecha abajo, izquierda abajo y cierra el cuadrado
cambio_coord_mov(150,260,-10)

#-----dibuja casillas-----

cambio_coord_mov(220,50,-10)
cambio_coord_mov(220,50,-40)#se posiciona para marcar la primera linea
cambio_coord_mov(220,50,-10)
cambio_coord_mov(290,50,-10)
cambio_coord_mov(290,50,-40) #primera linea lista
cambio_coord_mov(290,50,-10)

cambio_coord_mov(220,120,-10)
cambio_coord_mov(220,120,-40)  #se posiciona para marcar la primera linea
cambio_coord_mov(220,120,-10)
cambio_coord_mov(290,120,-10)
cambio_coord_mov(290,120,-40)
cambio_coord_mov(220,50,-10)

cambio_coord_mov(220,190,-10)
cambio_coord_mov(220,190,-40)  #se posiciona para marcar la primera linea
cambio_coord_mov(220,190,-10)
cambio_coord_mov(290,190,-10)
cambio_coord_mov(290,190,-45)
cambio_coord_mov(290,190,-10)

cambio_coord_mov(220,260,-10)
cambio_coord_mov(220,260,-40)  #se posiciona para marcar la primera linea
cambio_coord_mov(220,260,-10)
cambio_coord_mov(290,260,-10)
cambio_coord_mov(290,260,-40)
cambio_coord_mov(290,260,-10)

cambio_coord_mov(150,120,-10)
cambio_coord_mov(150,120,-40)  #se posiciona para marcar la primera linea
cambio_coord_mov(150,120,-10)
cambio_coord_mov(360,120,-10)
cambio_coord_mov(360,120,-40)
cambio_coord_mov(360,120,-10)

cambio_coord_mov(150,190,-10)
cambio_coord_mov(150,190,-40)  #se posiciona para marcar la primera linea
cambio_coord_mov(150,190,-10)
cambio_coord_mov(360,190,-10)
cambio_coord_mov(360,190,-40)
cambio_coord_mov(360,190,-10)
