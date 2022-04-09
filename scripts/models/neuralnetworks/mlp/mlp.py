import torch
import torch.nn as nn
from scripts.models.neuralnetworks.neuralnetwork import NeuralNetwork

torch.set_num_threads(1)


class MLP(NeuralNetwork):
    def __init__(self, config: dict):
        super(MLP, self).__init__(config=config)
        self.nhidden = config["n_hidden"]
        hdim = config["hidden_dim"]
        self.inlinear = nn.Linear(config["input_dim"] * config["sequence_length"], hdim)
        for i in range(self.nhidden):
            vars(self)['_modules'][f"hl{i+1}"] = nn.Linear(hdim, hdim)
        self.outlinear = nn.Linear(hdim, config["output_dim"])
        self.dropout = nn.Dropout(p=config["dropout"])
        self.non_linearity = eval(config["non_linearity"])()

    def forward(self, x):
        """
        :param x: input vector
        """
        z = self.non_linearity(self.inlinear(x))
        z = self.dropout(z)
        z = self.non_linearity(self.hl1(z)) # two times (second in for loop)??
        z = self.dropout(z)
        for i in range(self.nhidden):
            z = self.non_linearity(vars(self)['_modules'][f"hl{i+1}"](z))
            z = self.dropout(z)
        out = self.outlinear(z)
        return out

    def predict(self, x):
        """
        :param x: input vector
        """
        return self.forward(x)
