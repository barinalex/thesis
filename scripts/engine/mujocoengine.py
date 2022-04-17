import mujoco_py
import os
import time
import numpy as np
from scripts.constants import Dirs, DT
import quaternion
from multiprocessing import Lock
from scripts.engine.engine import Engine
from scripts.utils.simplexactionnoise import SimplexNoise
from scripts.datamanagement.pathmanagement import gettimestamp, create_directories
from scripts.datamanagement.datamanagementutils import save_raw_data
mutex = Lock()

mj_path = mujoco_py.utils.discover_mujoco()
xml_path = os.path.join(mj_path, 'model', 'waypointsonecar.xml')


class MujocoEngine(Engine):
    def __init__(self, visualize: bool = False):
        super().__init__()
        self.model = mujoco_py.load_model_from_path(xml_path)
        self.sim = mujoco_py.MjSim(self.model, nsubsteps=1)
        self.bodyid = self.model.body_name2id('buddy')
        self.viewer = mujoco_py.MjViewerBasic(self.sim) if visualize else None
        self.start = time.time()
        self.wpi = 0
        self.n_wps = 10

    def reset(self):
        self.start = time.time()
        self.state.reset()
        self.sim.reset()

    def movewaypoint(self, pos: np.ndarray):
        """
        move waypoint body to new position

        :param pos: waypoint position shape (2,)
        """
        self.sim.data.set_mocap_pos(f"waypoint{self.wpi}", np.hstack((pos, [0])))
        self.wpi = (self.wpi + 1) % self.n_wps

    def movecar(self, pos: np.ndarray, euler: np.ndarray):
        """
        move car body to new position and orientation

        :param pos: waypoint position shape (2,)
        :param euler: orientation in euler angles, shape (3,)
        """
        # self.sim.data.set_mocap_pos("buddy", np.hstack((pos, [0])))
        pass

    def get_pos(self) -> np.ndarray:
        """
        :return: position of a center of mass
        """
        return self.sim.data.body_xpos[self.bodyid].copy()

    def get_orn(self) -> np.ndarray:
        """
        :return: orientation of a buggy
        """
        orn = self.sim.data.body_xquat[self.bodyid].copy()
        if (orn == np.zeros(4)).all():
            return quaternion.as_float_array(quaternion.one)
        return orn

    def get_map_vel(self) -> np.ndarray:
        """
        :return: linear velocity in a map frame
        """
        return self.sim.data.body_xvelp[self.bodyid].copy()

    def get_lin(self) -> np.ndarray:
        """
        :return: linear velocity in a robot frame
        """
        npq = self.get_orn()
        npq = np.array([1, 0, 0, 0]) if (npq == np.zeros(4)).all() else npq
        q = quaternion.quaternion(*npq)
        q.w = -q.w
        vel = self.get_map_vel()
        return quaternion.as_rotation_matrix(q) @ vel

    def get_ang(self) -> np.ndarray:
        """
        :return: angular velocity
        """
        return self.sim.data.body_xvelr[self.bodyid].copy()

    def step(self, throttle: float, turn: float):
        throttle = ((throttle + 1) / 2)
        self.sim.data.ctrl[:] = [turn, throttle]
        self.sim.forward()
        self.sim.step()
        self.state.set(pos=self.get_pos(), orn=self.get_orn(), vel=self.get_lin(), ang=self.get_ang())
        if self.viewer and time.time() - self.start < self.sim.data.time:
            self.viewer.render()

    def gatherdata(self, n_steps: int = 3200):
        """
        :param n_steps: number of timesteps per episode
        """
        # from scripts.simulation.joystickinputwrapper import JoystickInputWrapper
        # jw = JoystickInputWrapper()
        position = np.zeros((n_steps, 3))
        orientation = np.zeros((n_steps, 4))
        linear = np.zeros((n_steps, 3))
        angular = np.zeros((n_steps, 3))
        actions = np.zeros((n_steps, 2))

        throtlegenerator = SimplexNoise(dim=1, smoothness=300, multiplier=2)
        turngenerator = SimplexNoise(dim=1, smoothness=350, multiplier=2)
        start = time.time()
        for i in range(n_steps):
            throttle = float(throtlegenerator())
            throttle = np.random.choice([throttle, -1], p=[0.9, 0.1])
            turn = float(turngenerator())
            # action, x = jw.getinput()
            # if x:
            #     return
            position[i, ] = self.get_pos()
            orientation[i, ] = self.get_orn()
            linear[i, ] = self.get_lin()
            angular[i, ] = self.get_ang()
            actions[i, ] = [throttle, turn]

            self.step(throttle=throttle, turn=turn)

        with mutex:
            path = os.path.join(Dirs.simdata, gettimestamp())
            create_directories(path=path)
        save_raw_data(data=position, path=os.path.join(path, f"{DT.pos}.npy"))
        save_raw_data(data=orientation, path=os.path.join(path, f"{DT.orn}.npy"))
        save_raw_data(data=linear, path=os.path.join(path, f"{DT.lin}.npy"))
        save_raw_data(data=angular, path=os.path.join(path, f"{DT.ang}.npy"))
        save_raw_data(data=actions, path=os.path.join(path, f"{DT.act}.npy"))


if __name__ == "__main__":
    from scripts.simulation.joystickinputwrapper import JoystickInputWrapper
    iw = JoystickInputWrapper()
    eng = MujocoEngine(visualize=True)
    # for i in range(10):
    #     eng.movecar(pos=np.array([i, 0]), euler=np.array([0,0,i]))
    #     print(eng.getpos(), eng.getorn())
    #     eng.viewer.render()
    #     time.sleep(1)
    # for i in range(5):
    #     eng.gatherdata(n_steps=2000)
    #     eng.reset()
    # exit()
    interrupt = False
    while not interrupt:
        throttle, turn, interrupt = iw.getinput()
        eng.step(throttle=throttle, turn=turn)
        print(eng.get_lin())
