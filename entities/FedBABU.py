import torch

from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from entities.base import Client, Server


class FedBABUClient(Client):
    def __init__(self, client_id, args, train_set, test_set, global_test_id):
        super().__init__(client_id, args, train_set, test_set, global_test_id)

        self.clf_keys = None

    def train(self):
        """
        Perform local training using mini-batch SGD. Only representation layers are trained.
        :return: The update of model parameters.
        """
        self.model.train()

        train_loader = DataLoader(self.train_set, batch_size=self.args["batch_size"], shuffle=True)

        # -------------------Start the training of representation-------------------
        for name, param in self.model.named_parameters():
            if name in self.clf_keys:
                param.requires_grad = False
            else:
                param.requires_grad = True

        optimizer = SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.args["lr"],
                        weight_decay=self.args["weight_decay"], momentum=self.args["momentum"])

        for epoch in range(self.args["epochs"]):
            for _, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(device=self.args["device"]), labels.to(device=self.args["device"])

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def fine_tune(self):
        """
        Fine-tune the entire model (i.e., both representation and classifier).
        """
        self.model.train()

        train_loader = DataLoader(self.train_set, batch_size=self.args["batch_size"], shuffle=True)

        for name, param in self.model.named_parameters():
            param.requires_grad = True

        optimizer = SGD(self.model.parameters(), lr=self.args["lr"], weight_decay=self.args["weight_decay"],
                        momentum=self.args["momentum"])

        for epoch in range(self.args["epochs"]):
            for _, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(device=self.args["device"]), labels.to(device=self.args["device"])

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


class FedBABUServer(Server):
    def __init__(self, args):
        super().__init__(args)

        self.clf_keys = None

    def aggregate_rep(self, clients):
        """
        Aggregate all clients' local representation parameters.
        """
        rep_params = [param for name, param in self.model.named_parameters() if name not in self.clf_keys]
        new_params = torch.zeros_like(parameters_to_vector(rep_params))
        total_size = 0
        for client in clients:
            client_size = self.client_data_size[client.id]
            total_size += client_size
            client_rep_params = [param for name, param in client.model.named_parameters() if name not in self.clf_keys]
            client_params = parameters_to_vector(client_rep_params)
            new_params += client_size * client_params

        new_params /= total_size
        new_params.to(self.args["device"])

        vector_to_parameters(new_params, rep_params)
