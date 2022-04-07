#!/usr/bin/env python3
import os
import datetime
import rospy
import message_filters as mf
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point, Quaternion, Vector3
from datareader.msg import ActionsStamped, Actions
import numpy as np
import tf
import tf2_ros
from threading import Lock
import time
import copy


DATA_DIR = "/home/barinale/Documents/thesis/thesis/data/real"


def create_directories(path: str):
    """
    create all directories in the path that do not exist

    :return: path
    """
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def replace_symbols(var: str, to_replace: str, replacement: str):
    """
    :return: string with replaced symbols
    """
    for symbol in to_replace:
        var = var.replace(symbol, replacement)
    return var


def gettimestamp():
    """
    :return: date and time as a string
    """
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


def get_static_tf(source_frame, target_frame):
    """
    :param source_frame: source frame
    :param target_frame: target frame
    :return: static transformation from source to target frame
    """
    tfbuffer = tf2_ros.Buffer()
    tf2_ros.TransformListener(tfbuffer)
    for i in range(100):
        print("lookup", i)
        try:
            trans = tfbuffer.lookup_transform(target_frame,
                                              source_frame,
                                              rospy.Time(0),
                                              rospy.Duration(0))
            return trans
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as err:
            rospy.logwarn_throttle(1, "ros_utils tf lookup could not lookup tf: {}".format(err))
            time.sleep(0.2)
            continue


def rotate_vector_by_quat(v: Vector3, q: Quaternion):
    """
    :param v: vector
    :param q: quaternion
    :return: rotated vector by quaternion
    """
    qm = tf.transformations.quaternion_matrix([q.x, q.y, q.z, q.w])[:3, :3]
    new_v = np.matmul(qm, np.array([v.x, v.y, v.z]))
    return Vector3(x=new_v[0], y=new_v[1], z=new_v[2])


class BuggyReader:
    def __init__(self):
        self.counter = 0
        self.gt_odom_list = []
        self.gt_odom_lock = Lock()
        self.gt_odom_msg = None

        self.act_list = []
        self.act_lock = Lock()
        self.act_msg = None

        rospy.init_node("buggyreader")
        rospy.Subscriber("/camera/odom/sample", Odometry, self.callback)
        rospy.Subscriber("/actions", Actions, self.actcallback)
        self.path = f"{DATA_DIR}/{gettimestamp()}"

    def callback(self, odom: Odometry):
        """
        :param odom: message containing odometry data
        """
        print("MESSAGE: ", self.counter)
        self.counter += 1
        self.gt_odom_list.append(odom)
        with self.act_lock:
            self.act_list.append(copy.deepcopy(self.act_msg))

    def actcallback(self, act: Actions):
        """
        :param act: message containing actions data
        """
        with self.act_lock:
            self.act_msg = act

    def rotate_twist(self, odom_msg):
        """
        Transform the twist in the odom message
        :param odom_msg: odometry message
        :return: transformed odometry message
        """
        odom_msg.twist.twist.linear = rotate_vector_by_quat(odom_msg.twist.twist.linear,
                                                            self.bl2rs.transform.rotation)
        odom_msg.twist.twist.angular = rotate_vector_by_quat(odom_msg.twist.twist.angular,
                                                             self.bl2rs.transform.rotation)
        odom_msg.pose.pose.position.x -= self.bl2rs.transform.translation.x
        odom_msg.pose.pose.position.y -= self.bl2rs.transform.translation.y
        odom_msg.pose.pose.position.z -= self.bl2rs.transform.translation.z
        return odom_msg

    def gather(self):
        """
        extract data from messages and save
        """
        rospy.spin()
        print("Gathered: {} action and {} odom messages".format(len(self.act_list), len(self.gt_odom_list)))
        # self.gt_odom_list = [self.rotate_twist(msg) for msg in self.gt_odom_list]
        n = np.minimum(len(self.act_list), len(self.gt_odom_list))
        messages = [{"act_msg": self.act_list[i], "odom_msg": self.gt_odom_list[i]} for i in range(n)]
        self.extract_data(messages)

    def extract_data(self, messages):
        """
        :param messages: list of dictionaries with act and odom messages
        """
        n = len(messages)
        positions = np.zeros((n, 3))
        actions = np.zeros((n, 2))
        linvels = np.zeros((n, 3))
        angvels = np.zeros((n, 3))
        for i in range(n):
            act = messages[i]["act_msg"]
            odom = messages[i]["odom_msg"]
            positions[i] = [odom.pose.pose.position.x, odom.pose.pose.position.y, odom.pose.pose.position.z]
            actions[i] = [act.throttle, act.turn]
            linvels[i] = [odom.twist.twist.linear.x, odom.twist.twist.linear.y, odom.twist.twist.linear.z]
            angvels[i] = [odom.twist.twist.angular.x, odom.twist.twist.angular.y, odom.twist.twist.angular.z]
        create_directories(self.path)
        np.save(f"{self.path}/positions.npy", positions)
        np.save(f"{self.path}/actions.npy", actions)
        np.save(f"{self.path}/linear.npy", linvels)
        np.save(f"{self.path}/angular.npy", angvels)


if __name__ == "__main__":
    reader = BuggyReader()
    reader.gather()

