import os.path

import numpy as np
from scripts.models.model import Model
from scripts.models.neuralnetworks.mlp import MLP
from scripts.models.neuralnetworks.trainer import Trainer
from scripts.datamanagement.datamanagement import loadconfig, get_data
from scripts.datamanagement.utils import save_raw_data
from scripts.constants import Dirs
import torch


class MLPModel(Model):
    def __init__(self):
        super().__init__()
        self.model = None
        self.config = None

    def train(self, config: dict, savepath: str = None, finetune: bool = False):
        """
        Train new model and save parameters

        :param config: training configuration
        :param savepath: path to an existing directory with file name
        :param finetune: if True do not initialize new model
        """
        if not finetune:
            self.model = MLP(config=config)
        train, test, normcnst = get_data(params=config)
        config["normcnst"] = normcnst
        trainer = Trainer(train=train, test=test, model=self.model)
        evals, _ = trainer.train()
        if savepath:
            trainer.savemodel(path=f"{savepath}.params")
            save_raw_data(data=evals, path=f"{savepath}.evals")

    def load(self, path):
        """
        Load model parameters

        :param path: path to an existing model parameters and config files.
        """
        self.config = loadconfig(f"{path}.yaml")
        self.model = MLP(config=self.config)
        self.model.load_state_dict(state_dict=torch.load(f"{path}.params"))

    def predict(self, obs: np.ndarray) -> (np.ndarray, np.ndarray):
        obs = torch.tensor(obs, dtype=torch.float32)
        prediction = self.model.predict(x=obs)
        prediction = prediction.detach().numpy()
        normcnst = np.asarray(self.config["normcnst"])
        prediction = np.multiply(prediction, normcnst)
        return prediction[:2], prediction[2]


if __name__ == "__main__":
    model = MLPModel()
    model.load(os.path.join(Dirs.models, "mlp_2022_04_09_17_00_57_178774"))
