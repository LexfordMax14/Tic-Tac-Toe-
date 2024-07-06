import cv2
import numpy as np
import time
from mlf_api import RobotClient
from random import randint, choice

# Conexión con el robot
robot = RobotClient("192.168.0.107")

# Función para mostrar una imagen
def show(frame):
    cv2.imshow("Tablero", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Función para mostrar los contornos detectados en la imagen
def show_contours(frame, contours):
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
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

        # Toma una porción de la imagen utilizando ancho y alto
        h, w = image.shape
        image = image[0:int(h * 0.7), int(w * 0.25):int(w * 0.7)]

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
                print(f"Line: ({x1}, {y1}) -> ({x2, y2})")

        # Encuentra contornos
        contours, hierarchy = cv2.findContours(cv2.cvtColor(line_only_image, cv2.COLOR_BGR2GRAY), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(line_only_image, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Recorta la imagen a la caja delimitadora externa
        x, y, w, h = cv2.boundingRect(contours[0])
        line_only_image_crop = line_only_image[y:y + h, x:x + w]

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
                board_status.append(1)  # X
            elif occupied_o:
                board_status.append(0)  # O
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
    cambio_coord_mov(185, 225, -25)
    return None

def marca_casilla_8():
    cambio_coord_mov(290, 190, 20)
    cambio_coord_mov(255, 225, -25)
    return None

def marca_casilla_9():
    cambio_coord_mov(290, 190, 20)
    cambio_coord_mov(325, 225, -25)
    return None

# Implementación del algoritmo MiniMax

inf = float('infinity')

class Node:
    def __init__(self, value, parent=None):
        self.value = value
        self.parent = parent
        self.leaf = False
        self.score = None
        self.children = []

    def add_child(self, node):
        self.children.append(node)

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
        choice = None
        for child in self.root.children:
            if child.score == self.root.score:
                choice = child
                break
        if choice is None:
            return None
        state = self.root.value
        next_state = choice.value
        for i in range(9):
            if state[i] != next_state[i]:
                return i
        raise Exception('States are equal or different in size')

def winner(state, player):
    winning_combinations = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8],
        [0, 3, 6], [1, 4, 7], [2, 5, 8],
        [0, 4, 8], [2, 4, 6]
    ]
    for combination in winning_combinations:
        if all(state[pos] == player for pos in combination):
            return True
    return False

def terminal(state):
    return winner(state, 1) or winner(state, 0) or state.count(None) == 0

def next_states(state, turn):
    indices = [i for i, x in enumerate(state) if x is None]
    if not terminal(state):
        return [state[:i] + [turn] + state[i + 1:] for i in indices]
    return []

# Función para jugar al gato
def jugar_gato():
    try:
        # Inicializa el tablero
        board_status = [None] * 9

        # Iteración del juego hasta que haya un ganador o empate
        while not terminal(board_status):
            # Captura y procesamiento del estado actual del tablero
            board_status = estatus_tablero()
            if terminal(board_status):
                break

            # Crea el árbol MiniMax y decide el mejor movimiento
            tree = MiniMaxTree(Node(board_status))
            best_move = tree.best_action

            # Mueve el robot según el mejor movimiento
            if best_move == 0:
                marca_casilla_1()
            elif best_move == 1:
                marca_casilla_2()
            elif best_move == 2:
                marca_casilla_3()
            elif best_move == 3:
                marca_casilla_4()
            elif best_move == 4:
                marca_casilla_5()
            elif best_move == 5:
                marca_casilla_6()
            elif best_move == 6:
                marca_casilla_7()
            elif best_move == 7:
                marca_casilla_8()
            elif best_move == 8:
                marca_casilla_9()

            # Pausa para permitir la visualización y ajuste
            time.sleep(5)

            # Captura y procesamiento del estado actual del tablero después del movimiento del robot
            board_status = estatus_tablero()
            if terminal(board_status):
                break

            # Espera la interacción del jugador y vuelve a escanear el tablero
            input("Presiona Enter después de que el jugador haya realizado su movimiento...")
            board_status = estatus_tablero()

    except Exception as e:
        print(f"Error en el juego: {e}")

    finally:
        robot.closeWebRTC()


# Función principal para iniciar el juego
if __name__ == "__main__":
    jugar_gato()
