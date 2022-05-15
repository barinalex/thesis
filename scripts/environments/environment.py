from gym import Env
from gym.spaces import Box
import numpy as np
from scripts.datamanagement.utils import load_raw_data
from scripts.trajectories.waypointer import Waypointer
from scripts.utils.queuebuffer import QueueBuffer
from scripts.engine.mujocoengine import MujocoEngine
from scripts.engine.modelbased import ModelBased
from scripts.engine.engine import Engine
from scripts.constants import Dirs
from multiprocessing import Lock
import os
mutex = Lock()


class Environment(Env):
    """abstract class for an agent following trajectory in 2 dimensions"""
    def __init__(self, config: dict, engine: Engine, random: bool = False):
        """
        :param config: environment configuration
        :param engine: engine instance to update agent states
        :param random: if true after reset new trajectory is chosen randomly,
            if false go sequentially over the list of trajectories
        """
        self.verbose = False
        self.config = config
        self.engine = engine
        self.maxsteps = self.config["timelimit"]
        self.n_observations = self.config["n_observations"]
        self.n_waypoints = self.config["n_waypoints"]
        self.deviationthreshold = self.config["deviationthreshold"]
        self.maxangledeviation = np.pi
        points = load_raw_data(os.path.join(Dirs.trajectories, self.config["trajectories"]))
        self.waypointer = Waypointer(n_wps=self.n_waypoints, points=points, random=random)
        self.action_space = Box(low=-np.ones(2, dtype=np.float32),
                                high=np.ones(2, dtype=np.float32),
                                dtype=np.float32)
        obsmax = np.ones((self.n_observations, 3), dtype=np.float32)
        obsmax[:, :3] *= 10
        wpsmax = np.ones((self.n_waypoints, 2), dtype=np.float32).flatten() * 5
        self.observation_space = Box(low=np.hstack((-obsmax.flatten(), -wpsmax)),
                                     high=np.hstack((obsmax.flatten(), wpsmax)),
                                     dtype=np.float32)
        self.lastnorm = self.computenormtothegoal()
        self.lastdeviation = 0
        self.wpclosed = False
        self.stepsleft = self.maxsteps
        initvector = np.zeros(3)
        self.buffer = QueueBuffer(size=self.n_observations, initvector=initvector)
        self.initialize_waypoints_visualization()

    def initialize_waypoints_visualization(self):
        """
        Move mujoco objects to indicate trajectory
        """
        wps = self.waypointer.get_waypoints_vector()
        for wp in wps:
            self.engine.movewaypoint(wp)

    def move_waypoint(self):
        """
        Move mujoco objects to indicate trajectory
        """
        wps = self.waypointer.get_waypoints_vector()
        n_wps = self.engine.n_wps
        self.engine.movewaypoint(wps[n_wps - 1])

    def get_trajectory_direction(self):
        """
        :return: trajectory direction vector in the robot frame
            (difference between next two waypoint vectors)
        """
        waypoints = self.waypointer.get_waypoints_vector()
        waypoints = self.engine.toselfframe(np.hstack((waypoints[:2], np.zeros((2, 1)))))[:, :2]
        return waypoints[1] - waypoints[0]

    def get_cartotraj_angle(self):
        """
        :return: angle between the car x axis
            and the trajectory (line formed by two next waypoints)
        """
        direction = self.get_trajectory_direction()
        return abs(np.arctan2(direction[1], direction[0]))

    def get_veltotraj_angle(self):
        """
        :return: angle between the car velocity vector [Vx,Vy]
            and the trajectory (line formed by two next waypoints)
        """
        direction = self.get_trajectory_direction()
        lin = self.engine.getlin()[:2]
        carangle = np.arctan2(direction[1], direction[0])
        velangle = np.arctan2(lin[1], lin[0])
        return abs(velangle - carangle)

    def get_driving_angles(self) -> (float, float, float):
        """
        :return: (alpha, beta, gamma), where:
            alpha is an angle between the car velocity vector [Vx,Vy]
            and the car x axis;
            beta is an angle between the car x axis
            and the trajectory (line formed by two next waypoints)
            gamma is an angle between the car velocity vector [Vx,Vy]
            and the trajectory;
        """
        direction = self.get_trajectory_direction()
        lin = self.engine.getlin()[:2]
        alpha = np.arctan2(lin[1], lin[0])
        beta = np.arctan2(direction[1], direction[0])
        gamma = alpha - beta if np.linalg.norm(lin) > 0.05 else 0
        return abs(alpha), abs(beta), abs(gamma)

    def getgoal(self) -> np.ndarray:
        """
        :return: next waypoint
        """
        return self.waypointer.next_unvisited_point()

    def compute_reward(self, action) -> float:
        """
        :param action: last action agent has taken
        :return: reward for transition from state s_t
            to a state s_t+1 with an action a_t
        """
        reward = 1 if self.wpclosed else 0
        return reward
        # pos = self.engine.getpos()[:2]
        # deviation = self.waypointer.distance_to_trajectory(pos=pos)
        # a, b, g = self.get_driving_angles()
        # dpenalty = deviation * 0.01
        # apenalty = g * 0.01
        # return reward - dpenalty - apenalty

    def computenormtothegoal(self):
        """
        :return: norm between agent pos and the next waypoint
        """
        return np.linalg.norm(self.waypointer.next_unvisited_point() - self.engine.getpos()[:2])

    def make_observation(self) -> np.ndarray:
        """
        :return: agent's state observation
        """
        lin = self.engine.getlin()[:2]
        ang = self.engine.getang()[2]
        obs = np.hstack((lin, ang))
        self.buffer.add(element=obs)
        state = self.buffer.get_vector()
        wps = self.waypointer.get_waypoints_vector()
        wps = np.hstack((wps, np.zeros((wps.shape[0], 1))))
        wps = self.engine.toselfframe(vector=wps)
        return np.hstack((state, wps[:, :2].flatten()))

    def countsteps(self) -> bool:
        """
        :return: true if maxsteps threshold achieved
        """
        self.stepsleft -= 1
        return self.stepsleft <= 0

    def isdone(self) -> bool:
        """
        :return: true if time is out or deviation is unacceptable or drift angle is too high
        """
        done = self.countsteps()
        # pos = self.engine.getpos()[:2]
        # deviation = self.waypointer.distance_to_trajectory(pos=pos)
        # a, b, g = self.get_driving_angles()
        # done = done or g > self.maxangledeviation
        # done = done or deviation > self.deviationthreshold
        return done

    def updatetrajectory(self) -> None:
        """
        Update trajectory state, mark waypoint closure,
        update distance to the next waypoint and
        deviation from the trajectory
        """
        self.wpclosed = False
        pos = self.engine.getpos()[:2]
        if self.waypointer.update(pos=pos):
            self.wpclosed = True
            self.move_waypoint()
            self.lastnorm = self.computenormtothegoal()
            self.lastdeviation = self.waypointer.distance_to_trajectory(pos=pos)

    def step(self, action: np.ndarray):
        """
        Update engine and trajectory states, compute reward,
        make observation and check episode end condition

        :param action: [throttle, turn]
        """
        self.lastnorm = self.computenormtothegoal()
        pos = self.engine.getpos()[:2]
        self.lastdeviation = self.waypointer.distance_to_trajectory(pos=pos)
        self.engine.step(throttle=action[0], turn=action[1])
        self.updatetrajectory()
        # print(self.waypointer.next_unvisited_point(), self.engine.getpos())
        observation = self.make_observation()
        reward = self.compute_reward(action=action)
        done = self.isdone()
        # if done:
        #     reward -= self.stepsleft * 2
        return observation, reward, done, {}

    def render(self):
        pass

    def reset(self):
        """reset agent state, trajectory, timer etc."""
        with mutex:
            self.engine.reset()
            self.waypointer.reset()
            self.initialize_waypoints_visualization()
        self.stepsleft = self.maxsteps
        self.lastnorm = self.computenormtothegoal()
        self.lastdeviation = 0
        return self.make_observation()


if __name__ == "__main__":
    from scripts.datamanagement.datamanagement import loadconfig
    from scripts.simulation.joystickinputwrapper import JoystickInputWrapper
    from scripts.engine.tcnnbased import TCNNBased
    from scripts.engine.mlpbased import MLPBased
    from scripts.engine.mujocoengine import MujocoEngine
    iw = JoystickInputWrapper()
    config = loadconfig(os.path.join(Dirs.configs, "env.yaml"))
    # path = os.path.join(Dirs.models, "mlp_2022_05_01_12_30_00_981419")
    # engine = TCNNBased(path=path, visualize=True)
    engine = MujocoEngine(visualize=True)
    # engine = MLPBased(path=path, visualize=True)
    # config["trajectories"] = "lap_pd02_r1_s2.npy"
    env = Environment(config=config, engine=engine, random=True)
    interrupt = False
    done = False
    sumrewards = 0
    while not done and not interrupt:
        throttle, turn, interrupt = iw.getinput()
        obs, reward, done, _ = env.step(action=np.asarray([throttle, turn]))
        sumrewards += reward
        # print(throttle, turn, env.engine.getlin())
    print(sumrewards)
