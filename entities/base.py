import torch
import random
import numpy as np

from typing import Iterator
from torch.optim import SGD
from torch.nn import Parameter, CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from utils.models import get_model
from utils.metric import get_accuracy, save_results


class Client:
    def __init__(self, client_id, args, train_set, test_set, global_test_id):
        self.id = client_id
        self.args = args
        self.train_set = train_set
        self.test_set = test_set
        self.global_test_id = global_test_id
        self.model = get_model(args["dataset"], args["model"]).to(args["device"])
        self.criterion = CrossEntropyLoss().to(args["device"])
        self.local_update = None

    def set_params(self, new_params: Iterator[Parameter]):
        """
        Set local model's parameters as new parameters.
        :param new_params: The new parameters to be set.
        """
        for new_param, local_param in zip(new_params, self.model.parameters()):
            local_param.data = new_param.data.clone()

    def train(self):
        """
        Perform local training using mini-batch SGD.
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

        self.local_update = parameters_to_vector(self.model.parameters()) - init_params

    def local_test(self):
        local_accuracy = get_accuracy(self.model, self.test_set, self.args["device"])

        return local_accuracy

    def global_test(self, global_tests):
        global_accuracy = get_accuracy(self.model, global_tests[self.global_test_id], self.args["device"])

        return global_accuracy


class Server:
    def __init__(self, args):
        self.args = args
        self.model = get_model(args["dataset"], args["model"]).to(args["device"])
        self.client_data_size = []
        self.writer = SummaryWriter(
            f"../../runs/{args['client_num']}/{args['dataset']}_{args['drift_pattern']}/{args['partition']}/{args['model']}/{args['rounds']}"
        )

    def get_client_data_size(self, clients):
        """
        Get the dataset sizes of all clients, which are used to determine the weight of aggregation.
        :param clients: ALl clients.
        """
        self.client_data_size = [len(client.train_set) for client in clients]

    def select_clients(self, clients):
        selected_clients = random.sample(clients, int(self.args["client_num"] * self.args["sample_ratio"]))

        return selected_clients

    def send_params(self, clients):
        for client in clients:
            client.set_params(self.model.parameters())

    def aggregate_by_params(self, clients):
        """
        Aggregate all clients' local parameters.
        """
        new_params = torch.zeros_like(parameters_to_vector(self.model.parameters()))
        total_size = 0
        for client in clients:
            client_size = self.client_data_size[client.id]
            total_size += client_size
            client_params = parameters_to_vector(client.model.parameters())
            new_params += client_size * client_params

        new_params /= total_size
        new_params.to(self.args["device"])

        vector_to_parameters(new_params, self.model.parameters())

    def aggregate_by_updates(self, clients):
        """
        Aggregate all clients' local updates.
        """
        total_size = 0
        update_sum = torch.zeros_like(parameters_to_vector(self.model.parameters()))
        for client in clients:
            client_size = self.client_data_size[client.id]
            total_size += client_size
            update_sum += client_size * client.local_update

        aggregated_update = update_sum / total_size

        new_params = parameters_to_vector(self.model.parameters()) + aggregated_update

        vector_to_parameters(new_params, self.model.parameters())

    def local_evaluate(self, clients, cur_round):
        accuracy_list = []
        for client in clients:
            accuracy_list.append(client.local_test())

        # TensorBoard log
        self.writer.add_scalars(
            "local_accuracy_mean",
            {self.args["algorithm"]: np.array(accuracy_list).mean()},
            cur_round
        )
        local_accuracy = np.array(accuracy_list).mean()

        return local_accuracy

    def global_evaluate(self, clients, global_test_sets, cur_round):
        accuracy_list = []
        for client in clients:
            accuracy_list.append(client.global_test(global_test_sets))

        # TensorBoard log
        self.writer.add_scalars(
            "global_accuracy_mean",
            {self.args["algorithm"]: np.array(accuracy_list).mean()},
            cur_round
        )
        global_accuracy = np.array(accuracy_list).mean()

        return global_accuracy

    def last_round_evaluate(self, clients, global_test_sets):
        """
        Evaluate all clients' local accuracy and global accuracy, and save each client's metric.
        :param clients: All clients.
        :param global_test_sets: Full test sets.
        """
        info = []
        local_accuracy_list, global_accuracy_list = [], []
        for client in clients:
            local_accuracy_list.append(client.local_test())
            global_accuracy_list.append(client.global_test(global_test_sets))

        for index, client in enumerate(clients):
            self.args.update({
                "client_id": client.id,
                "local_accuracy": local_accuracy_list[index],
                "global_accuracy": global_accuracy_list[index]
            })
            info.append(self.args.copy())

        save_results(info, self.args)
