import os.path
# import Adafruit_PCA9685
import board
import busio
import adafruit_pca9685
import time
import numpy as np
from utils import loadconfig
import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.INFO)


class PWMDriver:
    def __init__(self, config: dict):
        """
        :param config: configuration dictionary
        """
        self.config = config
        pwm_freq = int(1. / self.config["update_period"])
        self.pulse_denominator = (1000000. / pwm_freq) / 4096.
        self.servo_ids = [0, 1]  # THROTTLE IS 0, STEERING is 1
        self.throttlemin = 0.5
        self.steeringmiddle = 0.5
        logging.info("Initializing the PWMdriver. ")
        i2c = busio.I2C(board.SCL, board.SDA)
        self.pwm = adafruit_pca9685.PCA9685(i2c)
        # self.pwm = Adafruit_PCA9685.PCA9685()
        self.pwm.set_pwm_freq(pwm_freq)
        self.arm_escs()
        logging.info("Finished initializing the PWMdriver. ")

    def write_servos(self, actions: list):
        """
        :param actions: [throttle, steering]
        """
        for sid in self.servo_ids:
            val = np.clip(actions[sid], 0, 1)
            pulse_length = ((val + 1) * 1000) / self.pulse_denominator
            self.pwm.set_pwm(sid, 0, int(pulse_length))

    def arm_escs(self):
        """
        Write the lowest value to the servos
        """
        time.sleep(0.1)
        logging.info("Setting escs to lowest value. ")
        self.write_servos([self.throttlemin, self.steeringmiddle])
        time.sleep(0.3)
        
    def stop(self):
        """
        Write zero throttle to the motor and turn front wheels so
            the car doesn't go straight full speed in case
        """
        logging.info("stop")
        self.write_servos([self.throttlemin, 0])
        time.sleep(0.3)
