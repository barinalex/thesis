from gym import Env
from gym.spaces import Box
import numpy as np
from scripts.datamanagement.datamanagementutils import load_raw_data
from scripts.trajectories.waypointer import Waypointer
from scripts.utils.queuebuffer import QueueBuffer
from scripts.engine.engine import Engine
from scripts.constants import Dirs
from multiprocessing import Lock
import os
mutex = Lock()


class Environment(Env):
    """abstract class for an agent following trajectory in 2 dimensions"""
    def __init__(self, config: dict, engine: Engine):
        """
        :param config: environment configuration
        :param engine: engine instance to update agent states
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
        self.waypointer = Waypointer(n_wps=self.n_waypoints, points=points)
        self.action_space = Box(low=-np.ones(2, dtype=np.float32),
                                high=np.ones(2, dtype=np.float32),
                                dtype=np.float32)
        obsmax = np.ones((self.n_observations, 5), dtype=np.float32)
        obsmax[:, :3] *= 10
        wpsmax = np.ones((self.n_waypoints, 2), dtype=np.float32).flatten() * 5
        self.observation_space = Box(low=np.hstack((-obsmax.flatten(), -wpsmax)),
                                     high=np.hstack((obsmax.flatten(), wpsmax)),
                                     dtype=np.float32)
        self.lastnorm = self.computenormtothegoal()
        self.lastdeviation = 0
        self.wpclosed = False
        self.stepsleft = self.maxsteps
        initvector = np.array([0, 0, 0, -1, 0])
        self.buffer = QueueBuffer(size=self.n_observations, initvector=initvector)

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
        gamma = alpha - beta
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

    def computenormtothegoal(self):
        """
        :return: norm between agent pos and the next waypoint
        """
        return np.linalg.norm(self.waypointer.next_unvisited_point() - self.engine.getpos()[:2])

    def make_observation(self, action) -> np.ndarray:
        """
        :param action: last taken action. [throttle, turn]
        :return: agent's state observation
        """
        lin = self.engine.getlin()[:2]
        ang = self.engine.getang()[2]
        obs = np.hstack((lin, ang, action[0], action[1]))
        self.buffer.add(element=obs)
        return self.buffer.get_vector()

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
        pos = self.engine.getpos()[:2]
        deviation = self.waypointer.distance_to_trajectory(pos=pos)
        done = self.countsteps()
        a, b, g = self.get_driving_angles()
        done = done or g > self.maxangledeviation
        done = done or deviation > self.deviationthreshold
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
            self.lastnorm = self.computenormtothegoal()
            self.lastdeviation = self.waypointer.distance_to_trajectory(pos=pos)

    def step(self, action: list):
        """
        Update engine and trajectory states, compute reward,
        make observation and check episode end condition
        """
        self.lastnorm = self.computenormtothegoal()
        pos = self.engine.getpos()[:2]
        self.lastdeviation = self.waypointer.distance_to_trajectory(pos=pos)
        self.engine.step(throttle=action[0], turn=action[1])
        self.updatetrajectory()
        print(self.waypointer.next_unvisited_point(), self.engine.getpos())
        observation = self.make_observation(action=action)
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
        self.stepsleft = self.maxsteps
        self.lastnorm = self.computenormtothegoal()
        self.lastdeviation = 0
        return self.make_observation(action=[-1, 0])


if __name__ == "__main__":
    from scripts.engine.mujocoengine import MujocoEngine
    from scripts.datamanagement.datamanagement import loadconfig
    from scripts.simulation.joystickinputwrapper import JoystickInputWrapper
    iw = JoystickInputWrapper()
    config = loadconfig(os.path.join(Dirs.configs, "env.yaml"))
    env = Environment(config=config, engine=MujocoEngine(visualize=True))
    interrupt = False
    while not interrupt:
        throttle, turn, interrupt = iw.getinput()
        env.step(action=[throttle, turn])
