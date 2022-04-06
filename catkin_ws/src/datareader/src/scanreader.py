#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import LaserScan
import numpy as np


class ScanReader:
    def __init__(self):
        self.counter = 0
        rospy.init_node("scanreader")
        rospy.Subscriber("scan", LaserScan, self.callback)
        self.pub = rospy.Publisher("scanfiltered", LaserScan, queue_size=10)
        rospy.spin()

    def filterscan(self, scan: np.ndarray, bound: int = 200, threshold: float = 0.001) -> np.ndarray:
        counter = 0
        index = 0
        for i in range(0, len(scan)):
            counter += 1
            if scan[i] == np.inf:
                if counter < bound:
                    for j in range(index, i):
                        scan[j] = np.inf
                counter = 0
                index = i

        for i in range(1, len(scan)):
            if scan[i] - scan[i-1] > threshold:
                scan[i-1] = np.inf
        return scan


    def callback(self, msg: LaserScan):
        """
        :param msg: message containing raw laser scan
        """
        ranges = np.asarray(msg.ranges)
        filtered = self.filterscan(scan=ranges)
        msg.ranges = filtered
        self.pub.publish(msg)


if __name__ == "__main__":
    sr = ScanReader()

