from mlf_api import RobotClient
import time as tiempo
import cv2
import numpy as np

# Conexión con el robot
robot = RobotClient("192.168.0.107")

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

        image = cv2.imread('frame.jpg', cv2.IMREAD_GRAYSCALE)

        image = cv2.imread('frame.jpg', cv2.IMREAD_GRAYSCALE)

        # Toma una porción de la imagen utilizando ancho y alto
        h, w = image.shape
        image = image[0:int(h*0.7), int(w*0.25):int(w*0.7)]

        '''
        Recorte la zona de la hoja primero para eliminar info extra
        Lo ideal es aislar primero cada color por hsv para reducir mas info!!!
        '''

        # Erosiona la imagen
        kernel = np.ones((5, 5), np.uint8)
        erosion = cv2.erode(image, kernel, iterations=1)

        # Aplica detección de bordes utilizando Canny
        edges = cv2.Canny(erosion, 100, 200)

        # Aplica la transformada de Hough para detectar líneas
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=50)

        # Crea una copia de la imagen original para dibujar líneas
        line_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        # Crea una imagen en blanco con las mismas dimensiones
        line_only_image = np.zeros_like(line_image)

        # Dibuja las líneas en la imagen
        if lines is not None:
            slopes = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                slope = (y2 - y1) / (x2 - x1 + 1e-6)
                slopes.append(slope)
                cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.line(line_only_image, (x1, y1), (x2, y2), (255, 255, 255), 5)
                print(f"Line: ({x1}, {y1}) -> ({x2}, {y2})")

        # Encuentra el índice de la línea horizontal (pendiente mínima)
        min_slope = min(slopes)
        min_slope_index = slopes.index(min_slope)

        # Encuentra el índice de la línea vertical (pendiente máxima)
        max_slope = max(slopes)
        max_slope_index = slopes.index(max_slope)

        # Convierte line_only_image a escala de grises
        line_only_image_gray = cv2.cvtColor(line_only_image, cv2.COLOR_BGR2GRAY)

        # Encuentra contornos
        contours, hierarchy = cv2.findContours(line_only_image_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(line_only_image, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Recorta la imagen a la caja delimitadora externa
        x, y, w, h = cv2.boundingRect(contours[0])
        line_only_image_crop = line_only_image[y:y+h, x:x+w]

        while True:
            # Muestra las imágenes resultantes
            cv2.imshow('canny', edges)
            cv2.imshow('image', line_image)
            cv2.imshow('line_only_image', line_only_image)
            cv2.imshow('line_only_image_crop', line_only_image_crop)

            # Espera para cerrar la ventana al presionar 'q'
            if cv2.waitKey() & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()

        # Recorta la imagen del área del tablero
        show(line_only_image_crop)


        # Convierte la imagen a BGR
        rgbImage = cv2.cvtColor(line_only_image_crop, cv2.COLOR_RGB2BGR)

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
    tiempo.sleep(2)
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
from tiempo_decorator import time
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

#Ejecución del juego

cambio_coord_mov(20,200,30)

if __name__ == '__main__':
    initial_board = estatus_tablero()
    g = TicTacToe(npc_policy=MiniMaxTree, board=initial_board)
    g.run()
