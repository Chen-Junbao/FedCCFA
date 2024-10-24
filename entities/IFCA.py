import torch
import numpy as np

from torch.nn.utils import parameters_to_vector, vector_to_parameters
from entities.base import Client, Server
from utils.metric import get_loss
from utils.models import get_model


class IFCAClient(Client):
    def __init__(self, client_id, args, train_set, test_set, global_test_id):
        super().__init__(client_id, args, train_set, test_set, global_test_id)

        self.cluster_identity = 0

    def clustering(self, global_models):
        """
        Calculate the cluster identity.
        :param global_models: All global models used to calculate the cluster identity.
        """
        loss_list = []
        for model in global_models:
            loss = get_loss(model, self.train_set, self.criterion, self.args["device"])
            loss_list.append(loss)

        self.cluster_identity = np.array(loss_list).argmin(0)

        self.set_params(global_models[self.cluster_identity].parameters())


class IFCAServer(Server):
    def __init__(self, args):
        super().__init__(args)

        self.global_models = [get_model(args["dataset"], args["model"]).to(args["device"])
                              for _ in range(self.args["cluster_num"])]

    def send_params(self, clients):
        for client in clients:
            client.set_params(self.global_models[client.cluster_identity].parameters())

    def aggregate_with_clustering(self, clients):
        """
        Aggregate clients' updates for each global model.
        """
        client_groups = {}
        for identity in range(len(self.global_models)):
            client_groups[identity] = []

        for client in clients:
            client_groups[client.cluster_identity].append(client)

        # multiple-model aggregation
        for identity, global_model in enumerate(self.global_models):
            if len(client_groups[identity]) == 0:
                # no client updates this global model
                continue

            # model averaging
            total_size = 0
            new_params = torch.zeros_like(parameters_to_vector(global_model.parameters()))
            for client in client_groups[identity]:
                client_size = self.client_data_size[client.id]
                total_size += client_size
                client_params = parameters_to_vector(client.model.parameters())
                new_params += client_size * client_params

            new_params /= total_size
            new_params.to(self.args["device"])

            vector_to_parameters(new_params, global_model.parameters())
