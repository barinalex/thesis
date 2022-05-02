import time
import numpy as np
# import smbus
from joycontroller import JoyController
from rsreader import RSReader
from pwmdriver import PWMDriver
from utils import loadconfig, save2json, gettimestamp, save_raw_data
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
        self.agent = AgentDriver()
        self.history = []
        logging.info(f"Controller initialized")
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
            laststate = self.history[-1]
            lin, ang = np.copy(laststate["lin"]), np.copy(laststate["ang"])
            self.agent.update(lin=lin, ang=ang)
            throttle, steering = self.agent.act()
        acttime = time.time() - start
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
            logging.info(f"state dict: {statedict}")
            self.history.append(copy.deepcopy(statedict))
            throttle, steering = self.get_actions()
            mthrottle, msteering = self.actions2motor(throttle=throttle, steering=steering)
            logging.info(f"motor throttle: {mthrottle}; motor steering: {msteering}")
            self.history[-1]["act"] = [throttle, steering]
            self.history[-1]["servos"] = [mthrottle, msteering]
            self.driver.write_servos(actions=[mthrottle, msteering])
            itertime = time.time() - start
            time.sleep(max(0, self.config["update_period"] - itertime))

    def list2dict(self) -> dict:
        """
        Convert list of dicts to dict of lists
        """
        keys, items = self.history[0].items()
        n = len(self.history)
        data = {key: np.zeros((n, *item.shape)) for key, item in zip(keys, items)}
        for i, entry in enumerate(self.history):
            for key, item in entry.items():
                data[key][i] = item
        return data

    def __exit__(self, exc_type, exc_value, traceback):
        self.driver.stop()
        path = os.path.join("data", gettimestamp())
        data = self.list2dict()
        for key, item in data.items():
            save_raw_data(data=item, path=os.path.join(path, key))
        # save2json(path=path, data=self.history)
            

if __name__ == "__main__":
    with Controller() as controller:
        controller.loop_control()
