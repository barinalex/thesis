import os

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SRC_DIR, os.pardir))
DATA_DIR = f"{ROOT_DIR}/data"

datatypes = {"pos": "position_rob",
             "orn": "rotation_rob",
             "vel": "velocity_glob",
             "ang": "angular_velocity_rob",
             "jact": "throttleturn",
             "mact": "action",
             "ts": "timestamp"}


class DT:
    """
    string data types container
    """
    pos = "position_rob"
    orn = "rotation_rob"
    vel = "velocity_glob"
    ang = "angular_velocity_rob"
    jact = "throttleturn"
    mact = "action"
    ts = "timestamp"
    obs = "obs"
    labels = "labels"
    typeslist = [pos, orn, vel, ang, jact]
    traintesttypes = [vel, ang, jact]
    model = "model"
    data = "data"
    params = "p"
    cnn = "cnn"
    rnn = "rnn"
    nn = "nn"
    imitator = "imitator"
    ecnn = "cnnengine"
    ernn = "rnnengine"
    enn = "nnengine"
    epb = "pbengine"
    emj = "mjengine"
    predictor = "predictor"
    basicenv = "basicenvironment"
    basicppo = "basicppo"
    basicsac = "basicsac"


class Dirs:
    """
    project data directories
    """
    source = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(source, os.pardir))
    data = root + "/data"
    datasets = data + "/sensors/datasets"
    agent = data + "/agent"
    gail = agent + "/gail"
    engine = data + "/engine"
    model = data + "/model"
    simulation = data + "/simulation"
    configs = root + "/configurations"
    env = data + "/environment"
    trajectories = data + "/trajectories"
    urdf = root + "/pybulletmodel/urdf"


if __name__ == "__main__":
    pass