import torch
from torch import nn
import numpy as np
from scripts.models.neuralnetworks.neuralnetwork import NeuralNetwork


def compute_out_size(length: int, kernelsize: int, stride: int) -> int:
    return (length - kernelsize) // stride + 1


class TemporalCNN(NeuralNetwork):
    def __init__(self, config: dict):
        super(TemporalCNN, self).__init__(config=config)
        self.output_dim = self.config["out_dim"]
        self.sequencelength = self.config["sequence_length"]
        n_channels = eval(self.config["n_channels"])
        kernel = eval(self.config["kernel"])
        stride = eval(self.config["stride"])
        dilation = eval(self.config["dilation"])
        self.nonlinearity = eval(self.config["non_linearity"])()
        self.conv1 = nn.Conv1d(in_channels=n_channels[0],
                               out_channels=n_channels[1],
                               kernel_size=kernel[0],
                               stride=stride[0])
        slength = compute_out_size(self.sequencelength, kernel[0], stride[0])
        self.conv2 = nn.Conv1d(in_channels=n_channels[1],
                               out_channels=n_channels[2],
                               kernel_size=kernel[1],
                               stride=stride[1])
        slength = compute_out_size(slength, kernel[1], stride[1])
        self.conv3 = nn.Conv1d(in_channels=n_channels[2],
                               out_channels=n_channels[3],
                               kernel_size=kernel[2],
                               stride=stride[2])
        slength = compute_out_size(slength, kernel[2], stride[2])
        self.lin1 = nn.Linear(n_channels[3]*slength, 3)

    def forward(self, x):
        print(x.shape)
        x = self.nonlinearity(self.conv1(x))
        print(x.shape)
        x = self.nonlinearity(self.conv2(x))
        print(x.shape)
        x = self.nonlinearity(self.conv3(x))
        print(x.shape)
        out = self.lin1(x.reshape(x.shape[0], x.shape[1] * x.shape[2]))
        print(out.shape)
        return out

    def predict(self, x):
        return self.forward(x)


if __name__ == "__main__":
    from scripts.datamanagement.datamanagement import loadconfig
    from scripts.constants import Dirs
    import os
    cnn = TemporalCNN(loadconfig(path=os.path.join(Dirs.configs, "cnn.yaml")))
    x = np.random.rand(4, 5, 10) # batch, channels, sequence size
    x = torch.tensor(x, dtype=torch.float32)
    x = cnn.forward(x=x)
    print(x)
