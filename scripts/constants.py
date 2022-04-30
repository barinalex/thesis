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
    data = root + "/data"
    realdata = data + "/real2"
    simdata = data + "/sim"
    datasets = data + "/datasets"
    policy = data + "/policy"
    models = data + "/models"
    configs = root + "/configs"
    trajectories = data + "/trajectories"
    urdf = root + "/pybulletmodel/urdf"


if __name__ == "__main__":
    pass