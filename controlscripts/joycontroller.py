import pygame
import sys
import logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)


class JoyController:
    def __init__(self):
        logging.info("Initializing joystick controller")
        pygame.init()
        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()
        logging.info(f"Initialized : {self.joystick.get_name()}")

    def get_joystick_input(self) -> (float, float, bool):
        """
        :return: throttle [-1, 1], steering [-1, 1], done
        """
        pygame.event.pump()
        steering, throttle = self.joystick.get_axis(3), self.joystick.get_axis(1)
        button_x = self.joystick.get_button(0)
        pygame.event.clear()
        return -throttle, -steering, button_x


if __name__ == "__main__":
    jc = JoyController()
    done = False
    while not done:
        t, s, done = jc.get_joystick_input()
        print(t, s)
