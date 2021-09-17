import cv2
import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
import glfw

light_ambient = [0.25, 0.25, 0.25]
light_position = [-10, 5, 0, 2]


def main():
    DISPLAY_WIDTH = 900
    DISPLAY_HEIGHT = 900

    if not glfw.init():
        return
    glfw.window_hint(glfw.VISIBLE, False)
    window = glfw.create_window(DISPLAY_WIDTH, DISPLAY_HEIGHT, "hidden window", None, None)
    if not window:
        glfw.terminate()
        return
    glfw.make_context_current(window)

    gluPerspective(90, (DISPLAY_WIDTH / DISPLAY_HEIGHT), 0.01, 12)
    glEnable(GL_TEXTURE_2D)
    glEnable(GL_DEPTH_TEST)
    glDepthFunc(GL_LEQUAL)

    glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient)
    glLightfv(GL_LIGHT0, GL_POSITION, light_position)
    glEnable(GL_LIGHT0)
    glEnable(GL_LIGHTING)

    glRotatef(-90, 1, 0, 0)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glRotatef(285, 0, 0, 1)
    glTranslatef(-5, -3, -2)

    glBegin(GL_QUADS)
    glColor3f(1, 0, 0)
    glVertex3f(2, 2, 0)
    glVertex3f(2, 2, 2)
    glVertex3f(2, 6, 2)
    glVertex3f(2, 6, 0)
    glEnd()

    image_buffer = glReadPixels(0, 0, DISPLAY_WIDTH, DISPLAY_HEIGHT, GL_RGB, GL_UNSIGNED_BYTE)
    image = np.frombuffer(image_buffer, dtype=np.uint8).reshape(DISPLAY_WIDTH, DISPLAY_HEIGHT, 3)

    cv2.imshow("debug", image)

    glfw.destroy_window(window)
    glfw.terminate()

    cv2.waitKey(-1)

if __name__ == '__main__':
    main()