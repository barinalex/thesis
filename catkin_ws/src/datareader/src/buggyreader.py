#!/usr/bin/env python3
import os
import datetime
import rospy
import message_filters as mf
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point, Quaternion, Vector3
import numpy as np
import time


DATA_DIR = "/home/barinale/Documents/thesis/simtoreal2d/data/real"


def create_directories(path: str):
    """create all directories in the path
    that do not exist and return path"""
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def replace_symbols(var: str, to_replace: str, replacement: str):
    """replace all required symbols in the string with desired"""
    for symbol in to_replace:
        var = var.replace(symbol, replacement)
    return var


def gettimestamp():
    """date and time to a string"""
    s = str(datetime.datetime.today())
    return replace_symbols(s, to_replace=' -:.', replacement='_')


def vector3tonumpy(v: Vector3) -> np.ndarray:
    """
    :param v: 3d vector ros
    :return: numpy array shape (3,)
    """
    return np.array([v.x, v.y, v.z])


def pointtonumpy(v: Point) -> np.ndarray:
    """
    :param v: ros Point message
    :return: numpy array shape (3,)
    """
    return np.array([v.x, v.y, v.z])


def quaterniontonumpy(q: Quaternion) -> np.ndarray:
    """
    :param q: rotation as a ros quaternion
    :return: quaternion x,y,z,w as a numpy array
    """
    return np.array([q.x, q.y, q.z, q.w])


class BuggyReader:
    def __init__(self):
        self.counter = 0
        n = 5000
        self.angular = np.zeros((n, 3))
        self.linear = np.zeros((n, 3))
        self.position = np.zeros((n, 3))
        self.orientation = np.zeros((n, 4))

        rospy.init_node("buggyreader")
        odom = mf.Subscriber("/camera/odom/sample", Odometry)
        gyro = mf.Subscriber("/camera/gyro/sample", Imu)
        accel = mf.Subscriber("/camera/accel/sample", Imu)
        ts = mf.ApproximateTimeSynchronizer([odom, gyro, accel], 1, 1)
        ts.registerCallback(self.callback)
        rospy.spin()

    def __enter__(self):
        pass

    def callback(self, odom: Imu, gyro: Imu, accel: Imu):
        """
        :param msg: message containing raw laser scan
        """
        print(f"\nMESSAGE{self.counter}")
        self.angular[self.counter] = vector3tonumpy(gyro.angular_velocity)
        self.linear[self.counter] = vector3tonumpy(accel.linear_acceleration)
        self.position[self.counter] = pointtonumpy(odom.pose.pose.position)
        self.orientation[self.counter] = quaterniontonumpy(odom.pose.pose.orientation)
        print("\n")
        self.counter += 1

    def __exit__(self):
        directory = f"{DATA_DIR}/{gettimestamp()}"
        create_directories(directory)
        print("SAVE TO", directory)
        np.save(f"{directory}/angular.npy", self.angular[:self.counter])
        np.save(f"{directory}/linear.npy", self.linear[:self.counter])
        np.save(f"{directory}/position.npy", self.position[:self.counter])
        np.save(f"{directory}/orientation.npy", self.orientation[:self.counter])


if __name__ == "__main__":
    reader = BuggyReader()

