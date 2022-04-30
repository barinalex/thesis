import os


class DT:
    """
    string data types container
    """
    pos = "positions"
    orn = "orientation"
    lin = "linear"
    ang = "angular"
    act = "actions"
    ts = "timestamp"
    obs = "obs"
    labels = "labels"
    typeslist = [pos, orn, lin, ang, act]
    traintesttypes = [lin, ang, act]


class Dirs:
    """
    project data directories
    """
    source = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(source, os.pardir))
    data = os.path.join(root, "data")
    realdata = os.path.join(data, "real")
    experiments = os.path.join(data, "experiments")
    simdata = os.path.join(data, "sim")
    datasets = os.path.join(data, "datasets")
    policy = os.path.join(data, "policy")
    models = os.path.join(data, "models")
    configs = os.path.join(root, "configs")
    trajectories = os.path.join(data, "trajectories")
    urdf = os.path.join(root, "pybulletmodel", "urdf")


if __name__ == "__main__":
    pass