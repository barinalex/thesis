import numpy as np
import torch
import torch.nn as nn
from sklearn.utils import shuffle
from scripts.models.neuralnetworks.neuralnetwork import NeuralNetwork
from scripts.constants import DT

torch.set_num_threads(1)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')


class Trainer:
    def __init__(self, train: dict, test: dict, model: NeuralNetwork):
        self.model = model
        self.config = self.model.config
        self.obstrn, self.labelstrn = train[DT.obs], train[DT.labels]
        self.obstest, self.labelstest = test[DT.obs], test[DT.labels]
        self.criterion = nn.MSELoss()

    def train(self) -> (np.ndarray, NeuralNetwork):
        """
        define optimizer and run training of a neural network instance
        with evaluation on test subset after each epoch

        :return: (evaluations, trained model)
        """
        optimizer = torch.optim.Adam(self.model.parameters(),
                                     lr=self.config["learning_rate"],
                                     weight_decay=self.config["weight_decay"])
        n_epochs = self.config["num_epochs"]
        evaluations = np.zeros((n_epochs, 2))
        print(f"Training initialized for {n_epochs} epochs")
        for epoch in range(n_epochs):
            loss = None
            obstrn, labelstrn = shuffle(self.obstrn, self.labelstrn)
            for i in range(len(obstrn)):
                obs_batch = torch.from_numpy(obstrn[i]).to(device=device)
                labels_batch = torch.from_numpy(labelstrn[i]).to(device=device)
                prediction = self.model.predict(obs_batch.float())
                loss = self.criterion(prediction, labels_batch.float())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            evaluations[epoch] = [loss.item(), self.eval(self.model)]
            print(f"epoch: {epoch + 1}, trn_loss: {evaluations[epoch][0]}, tst_loss: {evaluations[epoch][1]}")
        return evaluations, self.model

    def eval(self, model):
        """
        make prediction on test subset and return test loss

        :param model: neural network that is being trained
        :return: test loss of trained a model
        """
        with torch.no_grad():
            obs = torch.from_numpy(self.obstest).to(device=device)
            labels = torch.from_numpy(self.labelstest).to(device=device)
            prediction = model.predict(obs.float())
            loss = self.criterion(prediction, labels.float())
        return loss.item()

    def savemodel(self, path):
        """
        Store model parameters to a file

        :param path: path to an existing directory with file name
        """
        torch.save(self.model.state_dict(), path)
