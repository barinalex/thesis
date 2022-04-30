import numpy as np
import pyrealsense2 as rs
import time
import quaternion
from utils import quat2euler
from threading import Lock
import sys
import logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)


class RSReader:
    def __init__(self):
        logging.info("Initializing the rs_t265.")
        self.rs_lock = Lock()
        self.rs_to_world_mat = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
        self.pipe = rs.pipeline()
        self.cfg = rs.config()
        self.cfg.enable_stream(rs.stream.pose)
        device = self.cfg.resolve(self.pipe).get_device()
        pose_sensor = device.first_pose_sensor()
        #pose_sensor.set_option(rs.option.enable_map_relocalization, 0)
        pose_sensor.set_option(rs.option.enable_pose_jumping, 0)
        #pose_sensor.set_option(rs.option.enable_motion_correction, 0)
        pose_sensor.set_option(rs.option.enable_relocalization, 0)
        self.pipe.start(self.cfg)
        self.pos = np.zeros(3)
        self.orn = quaternion.as_float_array(quaternion.one)
        self.euler = np.zeros(3)
        self.lin = np.zeros(3)
        self.ang = np.zeros(3)
        self.timestamp = time.time()
        logging.info("Finished initializing the rs_t265. ")

    def data2robotframe(self, data):
        """
        :param data: pose data from realsense
        :return: (position, orientation quaternion, orientation euler, linear velocity, angular velocity)
        """
        pos = np.array([data.translation.x, data.translation.y, data.translation.z])
        pos = self.rs_to_world_mat @ pos
        orientation = (data.rotation.w, data.rotation.z, data.rotation.x, data.rotation.y)
        euler = quat2euler(*orientation)
        ang = (data.angular_velocity.z, data.angular_velocity.x, data.angular_velocity.y)
        lin = np.array([data.velocity.x, data.velocity.y, data.velocity.z])
        rotation = quaternion.as_rotation_matrix(np.quaternion(*orientation))
        lin = self.rs_to_world_mat @ lin
        lin = np.matmul(rotation.T, lin)
        return pos, orientation, euler, lin, ang

    def update(self) -> dict:
        """
        Update data from realsense. In case of error return the last valid state
            along with the variable updated=False

        :return: dictionary: {position, orientation quaternion, orientation euler,
            linear velocity, angular velocity, timestamp: float, updated: bool}
        """
        self.timestamp = time.time()
        frames = self.pipe.wait_for_frames()
        pose = frames.get_pose_frame()
        updated = False
        if pose:
            self.pos, self.orn, self.euler, self.lin, self.ang = self.data2robotframe(data=pose.get_pose_data())
            updated = True
        return {"pos": np.copy(self.pos),
                "orn": np.copy(self.orn),
                "euler": np.copy(self.euler),
                "lin": np.copy(self.lin),
                "ang": np.copy(self.ang),
                "timestamp": self.timestamp,
                "updated": updated}

