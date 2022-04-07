import mujoco_py
import time
import os
import numpy as np
from

mj_path = mujoco_py.utils.discover_mujoco()
xml_path = os.path.join(mj_path, 'models', 'one_car.xml')


class MujocoVizualizer:
    def __init__(self, agent: BasicAgent):
        self.agent = agent
        # self.models = mujoco_py.load_model_from_path(xml_path)
        self.bodyid = agent.env.engine.bodyid # self.models.body_name2id('buddy')
        self.sim = agent.env.engine.sim #mujoco_py.MjSim(self.models, nsubsteps=1)
        self.viewer = mujoco_py.MjViewerBasic(self.sim)
        self.firstwaypoint = 0
        self.n_waypoints = 10
        self.joy = None
        # self.joy = JoystickInputWrapper()
        self.agent.env.waypointer.enable_mujoco_visualization(mujocovisualizer=self)

    def movewaypoint(self, pos: np.ndarray):
        """
        move waypoint body to new position

        :param pos: waypoint position shape (2,)
        """
        self.sim.data.set_mocap_pos(f"waypoint{self.firstwaypoint}", np.hstack((pos, [0])))
        self.firstwaypoint = (self.firstwaypoint + 1) % self.n_waypoints
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
        return self.sim.data.body_xquat[self.bodyid].copy()

    def get_map_vel(self) -> np.ndarray:
        """
        :return: linear velocity in a map frame
        """
        return self.sim.data.body_xvelp[self.bodyid].copy()

    def get_vel(self) -> np.ndarray:
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

    def step(self) -> (np.ndarray, bool):
        """update agent's state"""
        if self.joy:
            action, x = self.joy.getinput()
            action = np.array([action['throttle'], action['turn']])
        else:
            action = self.agent.act_on_observation(readstate=agent.env.readstate)
        _, _, done, _ = self.agent.env.step(action)
        # self.sim.data.ctrl[:] = [action[1], (action[0] + 1) / 2]
        # self.sim.forward()
        # self.sim.step()
        return action, done
        # simtime = time.time() - self.start
        # engtime = self.agent.env.engine.simtime
        # if simtime < engtime:
        #     time.sleep(engtime - simtime)

    def simulate(self):
        """run a simulation"""
        start = time.time()
        n_steps = 3000
        # positions = np.zeros((n_steps, 3))
        # orientations = np.zeros((n_steps, 4))
        # velocities = np.zeros((n_steps, 3))
        # angulars = np.zeros((n_steps, 3))
        # actions = np.zeros((n_steps, 2))
        # waypoints = np.zeros((n_steps, self.agent.env.n_waypoints * 2))

        self.agent.env.prolongepisode(n_steps=n_steps)
        for i in range(n_steps):
            # positions[i, ] = self.agent.env.engine.get_pos()
            # orientations[i, ] = self.agent.env.engine.get_orn()
            # velocities[i, ] = self.agent.env.engine.get_vel()
            # angulars[i, ] = self.agent.env.engine.get_ang()
            # wps = self.agent.env.trajectory.get_waypoints_vector()
            # wps = self.agent.env.engine.toselfframe(np.hstack((wps, np.zeros((len(wps), 1)))))[:, :2]
            # waypoints[i, ] = wps.flatten()

            action, done = self.step()
            # if done: break
            # actions[i, ] = action

            print(i, self.agent.env.episode_rewards[i], self.agent.env.engine.get_vel())
            if time.time() - start < self.agent.env.engine.simtime:
                self.viewer.render()
        # path = f"{Dirs.simulation}/{gettimestamp()}"
        # create_directories(path=path)
        # save_raw_data(data=positions, path=f"{path}/positions.npy")
        # save_raw_data(data=orientations, path=f"{path}/orientations.npy")
        # save_raw_data(data=velocities, path=f"{path}/velocities.npy")
        # save_raw_data(data=angulars, path=f"{path}/angulars.npy")
        # save_raw_data(data=actions, path=f"{path}/actions.npy")
        # save_raw_data(data=waypoints, path=f"{path}/waypoints.npy")
        print("SUM REWARD", self.agent.env.episode_rewards.sum())


if __name__ == "__main__":
    from scripts.agent.agents.basicppo import BasicPPO
    from scripts.agent.agents.basicsac import BasicSAC
    from scripts.agent.waypointer import Waypointer
    agent = BasicSAC(timestamp="2022_02_10_12_33_46_396882")
    agent.env.difficulty = 0.
    agent.env.reset()
    # agent.env.verbose = True
    # agent.env.trajectory = Trajectory2d(n_waypoints=agent.env.config['n_waypoints'], filenames="difficult0.npy", maxradius=3, minradius=0.4, pointsdistance=0.15, sectionlength=1)
    v = MujocoVizualizer(agent=agent)
    v.simulate()
