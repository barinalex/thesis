import torch
from torch import nn
import numpy as np
from scripts.models.neuralnetworks.neuralnetwork import NeuralNetwork


def compute_out_size(length: int, kernelsize: int, stride: int) -> int:
    return (length - kernelsize) // stride + 1


class TemporalCNN(NeuralNetwork):
    def __init__(self, config: dict):
        super(TemporalCNN, self).__init__(config=config)
        self.mtype = "cnn"
        self.input_dim = self.config["input_dim"]
        self.output_dim = self.config["output_dim"]
        n_channels = self.config["n_channels"]
        sequencelength = self.config["sequence_length"]
        kernel = eval(self.config["kernel"])
        stride = eval(self.config["stride"])
        self.convlayers = self.config["convlayers"]
        self.nonlinearity = eval(self.config["non_linearity"])()
        self.in1 = nn.Linear(self.input_dim, n_channels)
        # print(sequencelength)
        for i in range(self.convlayers):
            vars(self)['_modules'][f"cl{i + 1}"] = nn.Conv1d(in_channels=n_channels, out_channels=n_channels,
                                                             kernel_size=kernel[i], stride=stride[i])
            sequencelength = compute_out_size(length=sequencelength, kernelsize=kernel[i], stride=stride[i])
            # print(sequencelength)
        self.lin1 = nn.Linear(n_channels*sequencelength, 3)

    def forward(self, x):
        # MLP to map input to 32 dim
        x = self.nonlinearity(self.in1(x))
        # transpose to feed data to conv layers
        x = x.transpose(1, 2)
        # feed to conv layers
        for i in range(self.convlayers):
            x = self.nonlinearity(vars(self)['_modules'][f"cl{i+1}"](x))
        # flatten conv output and map to environment params
        out = self.lin1(x.reshape(x.shape[0], x.shape[1] * x.shape[2]))
        print(x.shape, out.shape)
        return out

    def predict(self, x):
        return self.forward(x)


if __name__ == "__main__":
    from scripts.datamanagement.datamanagement import loadconfig
    from scripts.constants import Dirs
    import os
    cnn = TemporalCNN(loadconfig(path=os.path.join(Dirs.configs, "cnn.yaml")))
    x = np.random.rand(4, 3, 5)
    x = torch.tensor(x, dtype=torch.float32)
    x = cnn.forward(x=x)
    print(x)
