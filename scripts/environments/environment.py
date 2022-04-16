from gym import Env
from gym.spaces import Box
import numpy as np
from scripts.datamanagement.pathmanagement import gettimestamp, create_directories
from scripts.datamanagement.datamanagement import loadconfig, saveconfig
from scripts.utils.queuebuffer import QueueBuffer
from scripts.trajectories.waypointer import Waypointer
from scripts.constants import Dirs, DT
from multiprocessing import Lock
mutex = Lock()


class Environment(Env):
    """abstract class for an agent following trajectory in 2 dimensions"""
    def __init__(self, config: dict):
        """
        :param config: environment configuration
        """
        self.verbose = False
        self.config = config
        self.engine = None
        self.maxsteps = self.config["timelimit"]
        self.n_waypoints = self.config["n_waypoints"]
        self.n_actions = self.config["n_actions"]
        try:
            filenames = eval(self.config["trajectory"])
        except:
            filenames = self.config["trajectory"]
        points = np.zeros(self.n_waypoints) # TODO load
        self.waypointer = Waypointer(n_wps=self.n_waypoints, points=points)
        self.action_space = Box(low=np.array([-1, -1], dtype=np.float32),
                                high=np.array([1, 1], dtype=np.float32),
                                dtype=np.float32)
        self.lastnorm = np.linalg.norm(self.waypointer.next_unvisited_point())
        self.lastdeviation = 0
        self.lastvel = 0
        self.stepsleft = self.maxsteps
        self.episode_rewards = np.zeros(self.maxsteps)

    def prolongepisode(self, n_steps: int):
        """
        :param n_steps: number of steps for on episode
        """
        self.maxsteps = n_steps
        self.stepsleft = self.maxsteps
        self.episode_rewards = np.zeros(self.maxsteps)

    def getgoal(self) -> np.ndarray:
        """return next waypoint"""
        return self.waypointer.next_unvisited_point()

    def compute_reward(self, action) -> float:
        """reward agent for coming closer to a waypoint
        and penalize for undesirable behavior"""
        return 0

    def computenormtothegoal(self):
        """compute norm between agent pos and the next waypoint"""
        return np.linalg.norm(self.waypointer.next_unvisited_point() - self.engine.getpos()[:2])

    def make_observation(self) -> np.ndarray:
        """return stacked state data and waypoints mapped to an agent frame"""
        pass

    def countsteps(self) -> bool:
        """
        :return: true if maxsteps threshold achieved
        """
        self.stepsleft -= 1
        return self.stepsleft <= 0

    def isdone(self) -> bool:
        """
        :return: true if an episode is over
        """
        return self.countsteps()

    def storeandreturnreward(self, action) -> float:
        """compute reward, save it in a buffer and return for farther usage"""
        reward = self.compute_reward(action=action)
        if 0 < self.stepsleft <= self.maxsteps:
            self.episode_rewards[self.maxsteps - self.stepsleft] = reward
        return reward

    def updatetrajectory(self) -> bool:
        """update trajectory with a new pos of an agent and
        compute a new norm in case of a waypoint change
        return true if old waypoint is passed"""
        if self.waypointer.update_points_state(self.engine.state.get_pos()[:2]):
            self.lastnorm = self.computenormtothegoal()
            self.lastdeviation = self.waypointer.distance_to_trajectory(pos=self.engine.get_pos()[:2])
            return True
        return False

    def step(self, action: list):
        """make action, compute reward, make observation,
        update goal and check episode end condition"""
        # self.actiontobuffer(action=action)
        # self.actions.add(element=action)
        self.lastnorm = self.computenormtothegoal()
        self.engine.step(action={"throttle": action[0], "turn": action[1]})
        reward = self.storeandreturnreward(action=action)
        observation = self.make_observation()
        self.updatetrajectory()
        self.lastvel = self.engine.state.get_vel()[0]
        done = self.isdone()
        # if done:
        #     reward -= self.stepsleft * 2
        return observation, reward, done, {}

    def render(self, mode):
        # self.engine.render()
        pass

    @abstractmethod
    def prereset(self):
        pass

    def reset(self):
        """reset agent state, create new trajectory to follow and reset timer"""
        self.engine.reset()
        self.waypointer.minradius = self.minradius
        self.waypointer.maxradius = self.maxradius
        self.waypointer.pointsdistance = self.pointsdistance
        self.waypointer.straightlength = self.sectionlength
        # self.trajectory.difficulty = self.difficulty
        self.waypointer.difficulty = np.random.rand() / 2
        self.waypointer.random = self.random
        with mutex:
            self.waypointer.reset()
        self.stepsleft = self.maxsteps
        self.episode_rewards = np.zeros(self.maxsteps)
        self.lastnorm = self.computenormtothegoal()
        # self.actions.reset()
        self.lastdeviation = 0
        self.lastvel = 0
        return self.make_observation()

    def get_episode_rewards(self) -> np.ndarray:
        """return all rewards agent got during the episode"""
        return self.episode_rewards

    def getrewards(self) -> float:
        """
        :return: sum of episode rewards
        """
        return self.episode_rewards.sum()

    def readstate(self):
        """
        :return: (pos: position vector, orn: a quaternion w x y z as a numpy array,
            vel: linear velocity vector, ang: angular velocity vector,
            waypoints: vector of trajectory points to follow)
        """
        return self.engine.state.get_pos(), self.engine.state.get_orn(), \
            self.engine.state.get_vel(), self.engine.state.get_ang(), \
            self.waypointer.get_waypoints_vector()
