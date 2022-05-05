import torch
from torch import nn
import numpy as np
from scripts.models.neuralnetworks.neuralnetwork import NeuralNetwork


def compute_out_size(length: int, kernelsize: int, stride: int, dilation: int) -> int:
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
                               stride=stride[0],
                               dilation=dilation[0])
        slength = compute_out_size(self.sequencelength, kernel[0], stride[0], dilation[0])
        print(slength)
        self.conv2 = nn.Conv1d(in_channels=n_channels[1],
                               out_channels=n_channels[2],
                               kernel_size=kernel[1],
                               stride=stride[1],
                               dilation=dilation[1])
        slength = compute_out_size(slength, kernel[1], stride[1], dilation[1])
        print(slength)
        self.conv3 = nn.Conv1d(in_channels=n_channels[2],
                               out_channels=n_channels[3],
                               kernel_size=kernel[2],
                               stride=stride[2],
                               dilation=dilation[2])
        slength = compute_out_size(slength, kernel[2], stride[2], dilation[2])
        print(slength)
        self.lin1 = nn.Linear(n_channels[3]*slength, 16)
        self.lin2 = nn.Linear(16, 3)

    def forward(self, x):
        x = self.nonlinearity(self.conv1(x))
        x = self.nonlinearity(self.conv2(x))
        x = self.nonlinearity(self.conv3(x))
        out = self.lin1(x.reshape(x.shape[0], x.shape[1] * x.shape[2]))
        out = self.nonlinearity(out)
        out = self.lin2(out)
        return out

    def predict(self, x):
        return self.forward(x)


if __name__ == "__main__":
    from scripts.datamanagement.datamanagement import loadconfig
    from scripts.constants import Dirs
    import os
    cnn = TemporalCNN(loadconfig(path=os.path.join(Dirs.configs, "cnn.yaml")))
    x = np.random.rand(4, 5, 20) # batch, channels, sequence size
    x = torch.tensor(x, dtype=torch.float32)
    x = cnn.forward(x=x)
    print(x)
