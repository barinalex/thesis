import os
from scripts.constants import Dirs
from scripts.models.mlpmodel import MLPModel
from scripts.models.cnnmodel import CNNModel
from scripts.datamanagement.datamanagement import loadconfig, saveconfig
from scripts.datamanagement.pathmanagement import gettimestamp, create_directories


def trainmlp():
    """
    Train new mlp model
    """
    model = MLPModel()
    path = os.path.join(Dirs.configs, "mlp.yaml")
    config = loadconfig(path=path)
    timestamp = gettimestamp()
    savepath = os.path.join(Dirs.models, f"mlp_{timestamp}")
    cfgpath = os.path.join(Dirs.models, f"mlp_{timestamp}.yaml")
    model.train(config=config, savepath=savepath)
    saveconfig(path=cfgpath, config=config)


def traincnn():
    """
    Train new cnn model
    """
    model = CNNModel()
    path = os.path.join(Dirs.configs, "cnn.yaml")
    config = loadconfig(path=path)
    timestamp = gettimestamp()
    savepath = os.path.join(Dirs.models, f"tcnn_{timestamp}")
    cfgpath = os.path.join(Dirs.models, f"tcnn_{timestamp}.yaml")
    model.train(config=config, savepath=savepath)
    saveconfig(path=cfgpath, config=config)


def finetunecnn():
    """
    Fine tune on real data tcnn trained on sim data
    """


if __name__ == "__main__":
    trainmlp()
    # traincnn()
