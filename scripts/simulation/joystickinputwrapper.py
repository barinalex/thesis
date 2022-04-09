import logging
import pygame
from scripts.simulation.inputwrapper import InputWrapper
import numpy as np


class JoystickInputWrapper(InputWrapper):
    def __init__(self):
        logging.info("Initializing joystick controller")
        pygame.init()
        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()
        logging.info("Initialized gamepad: {}".format(self.joystick.get_name()))
        logging.info("Finished initializing the joystick controller.")
        self.button_x_state = 0

    def getinput(self) -> (float, float, bool):
        pygame.event.pump()
        turn, throttle = [self.joystick.get_axis(3), self.joystick.get_axis(1)]
        button_x = self.joystick.get_button(1)
        pygame.event.clear()

        turn = -turn
        throttle = -throttle

        if self.button_x_state == 0 and button_x == 1:
            self.button_x_state = 1
            button_x = 1
        elif self.button_x_state == 1 and button_x == 0:
            self.button_x_state = 0
            button_x = 0
        elif self.button_x_state == 1 and button_x == 1:
            self.button_x_state = 1
            button_x = 0
        else:
            self.button_x_state = 0
            button_x = 0

        # correct joystick data
        if throttle < 0:
            throttle = 0
        if abs(turn) < 0.1:
            turn = 0

        throttle = (throttle - 0.5) * 2

        throttle = np.clip(throttle * 2, -1, 1)
        turn = np.clip(turn * 2, -1, 1)
        return throttle, turn, button_x
