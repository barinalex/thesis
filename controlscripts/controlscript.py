import time
import numpy as np
# import smbus
from joycontroller import JoyController
from rsreader import RSReader
from pwmdriver import PWMDriver
from utils import loadconfig, gettimestamp, save_raw_data, create_directories
import logging
import sys
import os.path
import copy
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
        logging.info(f"Controller initialized")
        self.agent = AgentDriver()
        logging.info(f"Policy loaded: {sys.getsizeof(self.agent.agent.policy)}")
        self.history = {"pos": [],
                        "orn": [],
                        "euler": [],
                        "lin": [],
                        "ang": [],
                        "timestamp": [],
                        "updated": [],
                        "act": [],
                        "servos": [],
                        "auto": [],
                        "acttime": []}
        logging.info(f"Initialize agent's policy by running it few times")
        for _ in range(5):
            self.agent.act()
        logging.info(f"Agent is prepared")

    def __enter__(self):
        return self

    def get_actions(self):
        """
        process input from the joystick. 
        if specified pass control to the AI agent.
        return actions: throttle, steering corresponding to the motors m1 and m2
        """
        start = time.time()
        throttle, steering, autonomous = self.JOYStick.get_joystick_input()
        if autonomous:
            lin, ang = np.copy(self.history["lin"][-1]), np.copy(self.history["ang"][-1])
            self.agent.update(lin=lin, ang=ang)
            throttle, steering = self.agent.act()
            throttle = (throttle + 1) / 2
        acttime = time.time() - start
        self.history["auto"].append(autonomous)
        self.history["acttime"].append(acttime)
        logging.info(f"autonomous: {autonomous}; throttle: {throttle}; steering: {steering}; acttime: {acttime}")
        return throttle, steering

    def actions2motor(self, throttle: float, steering: float) -> (float, float):
        """
        :param throttle: joystick action [-1, 1]
        :param steering: joystick action [-1, 1]
        :return: motor actions (throttle, steering)
        """
        throttle = 0.5 * throttle * self.config["motor_scalar"] + self.config["throttle_offset"]
        mthrottle = np.clip(throttle, 0.5, 1)
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
            statedict = self.rsreader.update()
            for key, item in statedict.items():
                self.history[key].append(item)
            throttle, steering = self.get_actions()
            mthrottle, msteering = self.actions2motor(throttle=throttle, steering=steering)
            self.history["act"].append([throttle, steering])
            self.history["servos"].append([mthrottle, msteering])
            self.driver.write_servos(actions=[mthrottle, msteering])
            itertime = time.time() - start
            time.sleep(max(0, self.config["update_period"] - itertime))

    def __exit__(self, exc_type, exc_value, traceback):
        self.driver.stop()
        path = os.path.join("data", gettimestamp())
        create_directories(path=path)
        for key, item in self.history.items():
            data = np.asarray(item)
            save_raw_data(data=data, path=os.path.join(path, key))
            

if __name__ == "__main__":
    with Controller() as controller:
        controller.loop_control()
