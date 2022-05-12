import os.path

import numpy as np
from scripts.models.model import Model
from scripts.models.neuralnetworks.cnn import TemporalCNN
from scripts.models.neuralnetworks.trainer import Trainer
from scripts.datamanagement.datamanagement import loadconfig, get_data
from scripts.datamanagement.utils import save_raw_data
from scripts.constants import Dirs, DT
import torch


class CNNModel(Model):
    def __init__(self):
        super().__init__()
        self.model = None
        self.config = None

    def reshape3ddata(self, config: dict, data: np.ndarray) -> np.ndarray:
        """
        :param config: cnn configuration
        :param data: array to reshape
        :return: reshaped data
        """
        dim1 = data.shape[0]
        dim2 = data.shape[1]
        dim3 = config["sequence_length"]
        dim4 = config["input_dim"]
        data = data.reshape((dim1, dim2, dim3, dim4))
        return np.transpose(data, (0, 1, 3, 2))

    def reshape2ddata(self, config: dict, data: np.ndarray) -> np.ndarray:
        """
        :param config: cnn configuration
        :param data: array to reshape
        :return: reshaped data
        """
        dim1 = data.shape[0]
        dim2 = config["sequence_length"]
        dim3 = config["input_dim"]
        data = data.reshape((dim1, dim2, dim3))
        return np.transpose(data, (0, 2, 1))

    def train(self, config: dict, savepath: str = None):
        """
        Train new model and save parameters

        :param config: training configuration
        :param savepath: path to an existing directory with file name
        """
        self.model = TemporalCNN(config=config)
        train, test, normcnst = get_data(params=config)
        train[DT.obs] = self.reshape3ddata(config=config, data=train[DT.obs])
        test[DT.obs] = self.reshape2ddata(config=config, data=test[DT.obs])
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
        self.model = TemporalCNN(config=self.config)
        self.model.load_state_dict(state_dict=torch.load(f"{path}.params"))

    def predict(self, obs: np.ndarray) -> (np.ndarray, np.ndarray):
        obs = torch.tensor(obs, dtype=torch.float32)
        prediction = self.model.predict(x=obs)
        prediction = prediction.detach().numpy()
        normcnst = np.asarray(self.config["normcnst"])
        prediction = np.multiply(prediction, normcnst)
        prediction = prediction.flatten()
        return prediction[:2], prediction[2]


if __name__ == "__main__":
    path = os.path.join(Dirs.configs, "cnn.yaml")
    config = loadconfig(path=path)
    model = CNNModel()
    model.train(config)
    # model.load(os.path.join(Dirs.models, "mlp_2022_04_09_17_00_57_178774"))
