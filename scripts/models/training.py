import os
from scripts.constants import Dirs
from scripts.models.mlpmodel import MLPModel
from scripts.models.cnnmodel import CNNModel
from scripts.datamanagement.datamanagement import loadconfig, saveconfig
from scripts.datamanagement.pathmanagement import gettimestamp, create_directories


def trainmlp(tag: str = ""):
    """
    Train new mlp model
    """
    model = MLPModel()
    path = os.path.join(Dirs.configs, "mlp.yaml")
    config = loadconfig(path=path)
    timestamp = gettimestamp()
    savepath = os.path.join(Dirs.models, f"mlp{tag}_{timestamp}")
    cfgpath = os.path.join(Dirs.models, f"mlp{tag}_{timestamp}.yaml")
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


def finetunemlp():
    """
    Fine tune on real data MLP trained on sim data
    """
    path = os.path.join(Dirs.models, "mlp_ft_2022_05_01_17_25_26_298465")
    model = MLPModel()
    model.load(path=path)
    path = os.path.join(Dirs.configs, "mlp.yaml")
    config = loadconfig(path=path)
    timestamp = gettimestamp()
    savepath = os.path.join(Dirs.models, f"mlp_ft_{timestamp}")
    cfgpath = os.path.join(Dirs.models, f"mlp_ft_{timestamp}.yaml")
    model.train(config=config, savepath=savepath, finetune=True)
    saveconfig(path=cfgpath, config=config)


if __name__ == "__main__":
    trainmlp(tag="_augmented")
    # traincnn()
    # finetunemlp()
