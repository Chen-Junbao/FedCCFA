import torch
import numpy as np

from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from entities.base import Client, Server
from utils.metric import get_loss


class FlashClient(Client):
    def __init__(self, client_id, args, train_set, test_set, global_test_id):
        super().__init__(client_id, args, train_set, test_set, global_test_id)

        self.loss_decrement = args["loss_decrement"]
        self.prev_val_loss = -1

    def train(self):
        """
        Perform early-stopping training.
        """
        self.model.train()

        train_loader = DataLoader(self.train_set, batch_size=self.args["batch_size"], shuffle=True)
        init_params = parameters_to_vector(self.model.parameters())

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

            val_loss = get_loss(self.model, self.test_set, self.criterion, self.args["device"])

            if self.prev_val_loss != -1:
                delta = self.prev_val_loss - val_loss
                if 0 < delta < self.loss_decrement / (epoch + 1):
                    break

            self.prev_val_loss = val_loss

        self.local_update = parameters_to_vector(self.model.parameters()) - init_params


class FlashServer(Server):
    def __init__(self, args):
        super().__init__(args)

        self.beta1 = args["beta1"]
        self.beta2 = args["beta2"]
        self.beta3 = 0
        self.tau = args["tau"]
        self.first_momentum = 0
        self.second_momentum = self.tau ** 2
        self.prev_second_momentum = 0
        self.delta_momentum = 0

    def aggregate_by_updates(self, clients):
        total_size = 0
        update_sum = torch.zeros_like(parameters_to_vector(self.model.parameters()))
        for client in clients:
            client_size = self.client_data_size[client.id]
            total_size += client_size
            update_sum += client_size * client.local_update

        aggregated_update = update_sum / total_size
        aggregated_update = aggregated_update.detach().cpu().numpy()

        self.first_momentum = self.beta1 * self.first_momentum + (1 - self.beta1) * aggregated_update
        self.prev_second_momentum = self.second_momentum
        self.second_momentum = self.beta2 * self.second_momentum + (1 - self.beta2) * np.square(aggregated_update)
        self.beta3 = np.abs(self.prev_second_momentum) / (
                np.abs(np.square(aggregated_update) - self.second_momentum) + np.abs(self.prev_second_momentum))
        self.delta_momentum = self.beta3 * self.delta_momentum + (1 - self.beta3) * (
                np.square(aggregated_update) - self.second_momentum)
        aggregated_update = self.args["server_lr"] * self.first_momentum / (
                np.sqrt(self.second_momentum) - self.delta_momentum + self.tau)

        cur_global_params = parameters_to_vector(self.model.parameters())
        new_global_params = cur_global_params + torch.tensor(aggregated_update).to(self.args["device"])

        vector_to_parameters(new_global_params, self.model.parameters())
