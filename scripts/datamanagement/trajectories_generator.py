import os
import time

import numpy as np
import quaternion
from scripts.engine.state import State2D

state = State2D(timestep=0.01)


def compute_velocities(points_distance: float, radius: float):
    """
    :param points_distance: distance between trajectory points in meters
    :param radius: radius of a curve in meters
    :return: (linear velocity m/s, angular velocity rad/s)
    """
    linear = np.array([points_distance / state.timestep, 0, 0])
    angular = np.array([0, 0, linear[0] / radius]) if radius != 0 else np.zeros(3)
    return linear, angular


def angular_motion(linear: np.ndarray, angular: np.ndarray, steps: int,
                   maxturn: float = 3 / 2 * np.pi, breakonmaxturn: bool = False,
                   reset: bool = False, dim: int = 2):
    """
    :param linear: linear velocity
    :param angular: angular velocity
    :param steps: number of steps to simulate
    :param maxturn: maximum turn angle allowed
    :param breakonmaxturn: if True return trajectory once maxturn achieved
    :param dim: dimensionality of a trajectory
    :param reset: if True reset engine (start at 0, 0)
    :return: a dim dimensional angular motion trajectory
    """
    state.reset() if reset else None
    state.set(vel=linear, ang=angular)
    trajectory = np.zeros((steps, dim))
    for i in range(0, steps):
        state.step()
        trajectory[i] = state.get_pos()[:dim]
        if abs(angular[2] * state.timestep * (i+1)) > maxturn:
            if breakonmaxturn:
                trajectory = trajectory[:i]
                break
            state.set(ang=np.zeros(3))
            maxturn = steps * np.pi
    return trajectory


def randomize_radius(minradius: float = 1.5, maxradius: float = 8, zeroprob: float = 0.1) -> float:
    """
    :param minradius: minimum radius of curve sections
    :param maxradius: maximum radius of curve sections
    :param zeroprob: probability of a zero radius (straight line)
    :return: random radius from the interval
    """
    radius = np.random.beta(a=1.5, b=5) * (maxradius - minradius) + minradius
    radius *= np.random.choice([0, 1], p=[zeroprob, 1 - zeroprob])
    return radius * np.random.choice([-1, 1])


def randomize_length(length: float = 10, n_sections: int = 4) -> np.ndarray:
    """
    :param length: length of a trajectory's section in meters
    :param n_sections: total number of straight and curve sections
    :return: randomized sections length
    """
    deviation = int(length / 3)
    np.random.seed(int(time.time() * 1000000) % 1000000)
    random_lengths = np.random.normal(loc=length, scale=deviation, size=n_sections)
    return np.clip(random_lengths, deviation, length + deviation)


def generate_curve(points_distance: float = 0.3, length: float = 1, radius: float = 0,
                   maxturn: float = 3/2*np.pi, breakonmaxturn: bool = False, reset: bool = False):
    """
    :param points_distance: distance between trajectory points
    :param length: length of a returned trajectory in meters
    :param radius: radius of a curve in meters
    :param maxturn: maximum total turn angle of a curve in radians
    :param breakonmaxturn: if True return trajectory once maxturn achieved
    :param reset: False if this trajectory will continue previous trajectory
    :return: curved line described by a set of points
    """
    linear, angular = compute_velocities(points_distance=points_distance, radius=radius)
    steps = int(length // (linear[0] * state.timestep))
    return angular_motion(linear=linear, angular=angular, steps=steps,
                          maxturn=maxturn, breakonmaxturn=breakonmaxturn, reset=reset)


def generate_random_trajectory(reset: bool = True,
                               points_distance: float = 0.15,
                               minradius: float = 0.5,
                               maxradius: float = 2.,
                               minturn: float = 1/2*np.pi,
                               maxturn: float = 2*np.pi,
                               straightprob: float = 0.005,
                               sectionlength: float = 1.5,
                               n_sections: int = 40) -> np.ndarray:
    """
    :param reset: False if this trajectory will continue previous trajectory
    :param points_distance: distance between trajectory points
    :param minradius: minimum radius of curve sections
    :param maxradius: maximum radius of curve sections
    :param minturn: minimum total turn angle of a curve in radians
    :param maxturn: maximum total turn angle of a curve in radians
    :param straightprob: probability of a straight line section
    :param sectionlength: length of a trajectory's section in meters
    :param n_sections: total number of straight and curve sections
    :return: trajectory described by a set of points
    """
    state.reset() if reset else None
    sectionlengths = randomize_length(length=sectionlength, n_sections=n_sections)
    # trajectory = np.empty((0, 2))
    trajectory = generate_curve(points_distance=points_distance, length=0.3, reset=False)
    for slength in sectionlengths:
        radius = randomize_radius(minradius=minradius, maxradius=maxradius, zeroprob=straightprob)
        curve = generate_curve(points_distance=points_distance, length=slength,
                               radius=radius, maxturn=maxturn, reset=False)
        trajectory = np.concatenate((trajectory, curve))
    return trajectory


def generate_circle(points_distance: float = 0.25, radius: float = 1):
    """
    :param points_distance: distance between trajectory points
    :param radius: radius of a curve in meters
    :return: circle trajectory
    """
    return generate_curve(points_distance=points_distance, radius=radius, length=10**5,
                          maxturn=np.pi * 2, breakonmaxturn=True)


def generate_infinity(points_distance: float = 0.25, radius: float = 1):
    """
    :param points_distance: distance between trajectory points
    :param radius: radius of a curve in meters
    :return: infinity trajectory
    """
    circle_left = generate_circle(points_distance=points_distance, radius=radius)
    state.set(orn=quaternion.as_float_array(quaternion.one))
    circle_right = generate_circle(points_distance=points_distance, radius=-radius)
    return np.concatenate((circle_left, circle_right))


def generate_loop(points_distance: float = 0.25, radius: float = 1, straight_length: float = 3):
    """
    :param points_distance: distance between trajectory points
    :param radius: radius of a curve in meters
    :param straight_length: length of a straight line section
    :return: circle trajectory
    """
    s0 = generate_curve(points_distance=points_distance, radius=0, length=straight_length)
    c0 = generate_curve(points_distance=points_distance, radius=radius, length=10**5,
                        maxturn=3/2*np.pi, breakonmaxturn=True)
    s1 = generate_curve(points_distance=points_distance, radius=0, length=straight_length)
    c1 = generate_curve(points_distance=points_distance, radius=-radius, length=10**5,
                        maxturn=3/2*np.pi, breakonmaxturn=True)
    c2 = generate_curve(points_distance=points_distance, radius=radius, length=10**5,
                        maxturn=3/2*np.pi, breakonmaxturn=True)
    s2 = generate_curve(points_distance=points_distance, radius=0, length=straight_length)
    c3 = generate_curve(points_distance=points_distance, radius=-radius, length=10**5,
                        maxturn=np.pi, breakonmaxturn=True)
    s3 = generate_curve(points_distance=points_distance, radius=0, length=straight_length + 0.3)
    c4 = generate_curve(points_distance=points_distance, radius=-radius, length=10**5,
                        maxturn=np.pi / 4, breakonmaxturn=True)
    return np.concatenate((s0, c0, s1, c1, c2, s2, c3, s3, c4,))


def generate_single_turn(points_distance: float = 0.15, radius: float = 1, straight_length: float = 1) -> np.ndarray:
    """
    :param points_distance: distance between trajectory points
    :param radius: radius of a curve in meters
    :param straight_length: length of a straight line section
    :return: trajectory with one turn
    """
    s0 = generate_curve(points_distance=points_distance, radius=0, length=straight_length)
    c0 = generate_curve(points_distance=points_distance, radius=radius, length=10**5,
                        maxturn=np.pi, breakonmaxturn=True)
    q = quaternion.from_euler_angles(alpha_beta_gamma=0, beta=0, gamma=np.pi)
    state.set(orn=quaternion.as_float_array(q))
    s1 = generate_curve(points_distance=points_distance, radius=0, length=straight_length)
    return np.concatenate((s0, c0, s1,))


def generate_double_turn(points_distance: float = 0.15, radius: float = 1, straight_length: float = 1) -> np.ndarray:
    """
    :param points_distance: distance between trajectory points
    :param radius: radius of a curve in meters
    :param straight_length: length of a straight line section
    :return: trajectory with left then right turn or otherwise
    """
    s0 = generate_curve(points_distance=points_distance, radius=0, length=straight_length)
    c0 = generate_curve(points_distance=points_distance, radius=radius, length=10**5,
                        maxturn=np.pi, breakonmaxturn=True)
    q = quaternion.from_euler_angles(alpha_beta_gamma=0, beta=0, gamma=np.pi)
    state.set(orn=quaternion.as_float_array(q))
    s1 = generate_curve(points_distance=points_distance, radius=0, length=straight_length)
    c1 = generate_curve(points_distance=points_distance, radius=-radius, length=10**5,
                        maxturn=np.pi, breakonmaxturn=True)
    q = quaternion.from_euler_angles(alpha_beta_gamma=0, beta=0, gamma=0)
    state.set(orn=quaternion.as_float_array(q))
    s2 = generate_curve(points_distance=points_distance, radius=0, length=straight_length)
    return np.concatenate((s0, c0, s1, c1, s2))


def generate_turn(reset: bool = True, points_distance: float = 0.15, difficulty: float = 1,
                  minradius: float = 0.2, maxradius: float = 2, straight_length: float = 1,
                  random: bool = True):
    """
    :param reset: False if this trajectory will continue previous trajectory
    :param points_distance: distance between trajectory points
    :param difficulty: describe how difficult trajectory should be. 1 is hard, 0 easy
    :param minradius: minimum possible radius of trajectory turns
    :param maxradius: maximum possible radius of trajectory turns
    :param straight_length: length of a straight line section
    :param random: if True first turn could be left or right with equal probability
    :return: trajectory with left then right turn or otherwise
    """
    state.reset() if reset else None
    radius = (maxradius - minradius) * (1 - difficulty) + minradius
    if random:
        radius *= np.random.choice([1, -1])
    return generate_double_turn(points_distance=points_distance,
                                radius=radius,
                                straight_length=straight_length)


def generate_n_one_turns(radiuses: np.ndarray, straight_lengths: np.ndarray, points_distance: float = 0.15) -> np.ndarray:
    """
    :param points_distance: distance between trajectory points
    :param radiuses: radiuses of a curve in meters (n, )
    :param straight_lengths: lengths of a straight line sections (n, )
    :return: trajectories with one turn
    """
    n = radiuses.shape[0]
    trajectories = []
    for i in range(n):
        state.reset()
        trajectories.append(generate_single_turn(points_distance=points_distance,
                                                 radius=radiuses[i],
                                                 straight_length=straight_lengths[i]))
    return np.asarray(trajectories)


def generate_lap(points_distance: float = 0.25, radius: float = 1, straight_length: float = 3):
    """
    :param points_distance: distance between trajectory points
    :param radius: radius of a curve in meters
    :param straight_length: length of a straight line section
    :return: lap trajectory
    """
    s0 = generate_curve(points_distance=points_distance, radius=0, length=straight_length)
    c0 = generate_curve(points_distance=points_distance, radius=radius, length=10**5,
                        maxturn=np.pi, breakonmaxturn=True)
    q = quaternion.from_euler_angles(alpha_beta_gamma=0, beta=0, gamma=np.pi)
    state.set(orn=quaternion.as_float_array(q))
    s1 = generate_curve(points_distance=points_distance, radius=0, length=straight_length)
    c1 = generate_curve(points_distance=points_distance, radius=radius, length=10**5,
                        maxturn=np.pi+0.1, breakonmaxturn=True)
    return np.concatenate((s0, c0, s1, c1))


def generate_random_trajectories(n: int = 100) -> np.ndarray:
    """
    :param n: number of random trajectories to generate
    :return: n random trajectories as a numpy array with shape (n, m, dim),
        where m is a number of waypoints in trajectory, dim is a dimensionality
    """
    trajectories = []
    minwaipoints = 10**9
    for i in range(n):
        trajectories.append(generate_random_trajectory(reset=True))
        minwaipoints = trajectories[i].shape[0] if trajectories[i].shape[0] < minwaipoints else minwaipoints
    np_trajectories = np.zeros((n, minwaipoints, 2))
    for i in range(n):
        np_trajectories[i] = trajectories[i][:minwaipoints]
    return np_trajectories


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from scripts.constants import Dirs
    from scripts.datamanagement.datamanagementutils import save_raw_data, load_raw_data
    n=100
    difficulties = np.random.rand(n) / 2 + 0.25
    i = 0
    for i in range(n):
        # d = difficulties[i]
        # t = generate_random_double_turn(difficulty=d)
        # save_raw_data(data=t, path=f"{Dirs.trajectories}/doubleturn_025_075_{i}.npy")
        t = load_raw_data(path=f"{Dirs.trajectories}/doubleturn_025_075_{i}.npy")
        plt.plot(t[:, 0], t[:, 1])
        i += 1
    plt.show()
    exit()
    rs = np.arange(3, 11) / 10
    print(rs)
    stls = np.ones(rs.shape[0])
    ts = generate_n_one_turns(radiuses=rs, straight_lengths=stls)
    print(ts[0].shape)
    for t in ts:
        plt.plot(t[:, 0], t[:, 1])
    plt.show()
    exit()
    n = 10
    # trajs = generate_random_trajectories(n=n)
    for i in range(n):
        # save_raw_data(data=trajs[i],path=f"{Dirs.trajectories}/spagetti{i}.npy")
        t = load_raw_data(path=f"{Dirs.trajectories}/spagetti{i}.npy")
        plt.plot(t[:, 0], t[:, 1])
    plt.show()
    exit()
    for t in trajs:
        plt.plot(t[:, 0], t[:, 1])
    plt.show()
