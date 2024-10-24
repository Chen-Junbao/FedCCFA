import numpy as np

from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.nn.utils import parameters_to_vector
from entities.base import Client, Server


class AdaptiveFedAvgClient(Client):
    def __init__(self, client_id, args, train_set, test_set, global_test_id):
        super().__init__(client_id, args, train_set, test_set, global_test_id)

        self.lr = args["lr"]

    def train(self):
        """
        Perform local training using mini-batch SGD and adaptive learning rate.
        """
        self.model.train()

        train_loader = DataLoader(self.train_set, batch_size=self.args["batch_size"], shuffle=True)
        init_params = parameters_to_vector(self.model.parameters())

        optimizer = SGD(self.model.parameters(), lr=self.lr, weight_decay=self.args["weight_decay"],
                        momentum=self.args["momentum"])

        for epoch in range(self.args["epochs"]):
            for _, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(device=self.args["device"]), labels.to(device=self.args["device"])

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        self.local_update = parameters_to_vector(self.model.parameters()) - init_params


class AdaptiveFedAvgServer(Server):
    def __init__(self, args):
        super().__init__(args)

        self.prev_mean = 0.0
        self.prev_mean_norm = 0.0
        self.prev_variance = 0.0
        self.prev_variance_norm = 0.0
        self.prev_ratio = 0.0
        self.beta1 = self.args["beta1"]
        self.beta2 = self.args["beta2"]
        self.beta3 = self.args["beta3"]
        self.client_init_lr = self.args["lr"]

    def cal_adaptive_lr(self, cur_round):
        cur_round += 1  # prevent division by 0
        cur_params = parameters_to_vector(self.model.parameters()).detach().cpu().numpy()

        mean = self.beta1 * self.prev_mean + (1 - self.beta1) * cur_params
        mean_norm = mean / (1 - pow(self.beta1, cur_round))

        variance = self.beta2 * self.prev_variance + (1 - self.beta2) * np.mean(
            (cur_params - self.prev_mean_norm) * (cur_params - self.prev_mean_norm))
        variance_norm = variance / (1 - pow(self.beta2, cur_round))

        if cur_round == 1:
            # no previous variance
            ratio = self.beta3 * self.prev_ratio + (1 - self.beta3)
        else:
            ratio = self.beta3 * self.prev_ratio + (1 - self.beta3) * (variance_norm / self.prev_variance_norm)
        ratio_norm = ratio / (1 - pow(self.beta3, cur_round))

        self.prev_mean = mean
        self.prev_mean_norm = mean_norm
        self.prev_variance = variance
        self.prev_variance_norm = variance_norm
        self.prev_ratio = ratio

        client_dynamic_lr = min(self.client_init_lr, self.client_init_lr * ratio_norm / cur_round)

        return client_dynamic_lr
