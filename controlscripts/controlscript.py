import time
import numpy as np
# import smbus
from joycontroller import JoyController
from rsreader import RSReader
from pwmdriver import PWMDriver
from utils import loadconfig, save2json, gettimestamp
import logging
import sys
import os.path
from agentdriver import AgentDriver
logging.basicConfig(stream=sys.stdout, level=logging.INFO)


class Controller:  
    def __init__(self):
        path = os.path.join("configs", "default.yaml")
        self.config = loadconfig(path=path)
        self.motors_on = self.config["motors_on"]
        logging.info(f"Initializing the Controller, motors_on: {self.config['motors_on']}")
        self.rsreader = RSReader()
        self.driver = PWMDriver(config=self.config)
        self.JOYStick = JoyController()
        self.agent = AgentDriver()
        self.history = []
        logging.info(f"Controller initialized")

    def __enter__(self):
        return self

    def get_actions(self):
        """
        process input from the joystick. 
        if specified pass control to the AI agent.
        return actions: throttle, steering corresponding to the motors m1 and m2
        """
        throttle, steering, autonomous = self.JOYStick.get_joystick_input()
        if autonomous:
            laststate = self.history[-1]
            lin, ang = np.copy(laststate["lin"]), np.copy(laststate["ang"])
            self.agent.update(lin=lin, ang=ang)
            throttle, steering = self.agent.act()
        logging.info(f"autonomous: {autonomous}; throttle: {throttle}; steering: {steering}")
        return throttle, steering

    def actions2motor(self, throttle: float, steering: float) -> (float, float):
        """
        :param throttle: joystick action [-1, 1]
        :param steering: joystick action [-1, 1]
        :return: motor actions (throttle, steering)
        """
        throttle = 0.5 * ((throttle + 1) / 2) * self.config["motor_scalar"]
        mthrottle = np.clip(throttle + self.config["throttle_offset"], 0.5, 1)
        msteering = (steering / 2) + 0.5
        return mthrottle, msteering

    def loop_control(self):
        """
        Get input from the joystick or an agent, send it to the motors,
            sleep to keep a correct update frequency
        """
        logging.info("Starting the control loop")
        while True:
            start = time.time()
            self.history.append(self.rsreader.update())
            throttle, steering = self.get_actions()
            mthrottle, msteering = self.actions2motor(throttle=throttle, steering=steering)
            self.history[-1]["act"] = [throttle, steering]
            self.history[-1]["servos"] = [mthrottle, msteering]
            self.driver.write_servos(actions=[mthrottle, msteering])
            itertime = time.time() - start
            time.sleep(max(0, self.config["update_period"] - itertime))

    def __exit__(self, exc_type, exc_value, traceback):
        self.driver.stop()
        path = os.path.join("data", gettimestamp())
        save2json(path=path, data=self.history)
            

if __name__ == "__main__":
    with Controller() as controller:
        controller.loop_control()
