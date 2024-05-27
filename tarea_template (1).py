# Mateo Arenas
# Lázaro Narváez
# Micael Covarrubias
from mlf_api import RobotClient
import time
import cv2

import numpy as np

def show(frame):
    cv2.imshow("XD", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def show_contours(frame, contours):
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    show(frame)

robot = RobotClient("twilight.local")
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

    ypx = (305-cx)*1.55
    xpx = (434-cy)*1.4
    x = 50*xpx/65
    print(x)
    y = 50*ypx/65
    print(y)
    z = 20
    offset = [65, 0, 75]
    q3 = 0
    robot.move_xyz(x, y, z, offset, q3)



except Exception as e: 
    print(e)

finally:
    robot.closeWebRTC()
"""
from matplotlib import pyplot as plt
import cv2
from mlf_api import RobotClient

robot = RobotClient("twilight.local")
robot.connectWebRTC()

frame = robot.get_frame()
plt.imshow(frame)
plt.show()
robot.closeWebRTC()
"""


#hola



