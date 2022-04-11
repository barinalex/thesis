import os
from scripts.constants import Dirs
from scripts.models.mlpmodel import MLPModel
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
    prmpath = os.path.join(Dirs.models, f"mlp_{timestamp}.params")
    cfgpath = os.path.join(Dirs.models, f"mlp_{timestamp}.yaml")
    model.train(config=config, savepath=prmpath)
    saveconfig(path=cfgpath, config=config)


if __name__ == "__main__":
    trainmlp()

