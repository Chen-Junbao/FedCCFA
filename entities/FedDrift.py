import torch
import numpy as np

from torch.nn.utils import parameters_to_vector, vector_to_parameters
from entities.base import Client, Server
from utils.metric import get_loss
from utils.models import get_model


class FedDriftClient(Client):
    def __init__(self, client_id, args, train_set, test_set, global_test_id):
        super().__init__(client_id, args, train_set, test_set, global_test_id)

        self.prev_train_set = train_set
        self.cluster_identity = 0

    def clustering(self, global_models):
        """
        Calculate the cluster identity. This method is used for clustered FL (IFCA, FedDrift, .etc.)
        When cluster identity is None, a new global model will be created.
        :param global_models: All global models used to calculate the cluster identity.
        """
        prev_loss_list = []
        loss_list = []

        # calculate the loss of each global model on the training dataset at the previous round
        for model in global_models:
            prev_loss_list.append(get_loss(model, self.prev_train_set, self.criterion, self.args["device"]))
        min_prev_loss = min(prev_loss_list)

        # calculate the loss of each global model on the training dataset at the current round
        for model in global_models:
            loss_list.append(get_loss(model, self.train_set, self.criterion, self.args["device"]))
        min_loss = min(loss_list)

        if min_loss > min_prev_loss + self.args["detection_threshold"]:
            # concept drift is detected, and create a new model for all drifted clients
            self.cluster_identity = None
        else:
            # select the best model from existing clusters
            self.cluster_identity = np.array(loss_list).argmin(0)


class FedDriftServer(Server):
    def __init__(self, args):
        super().__init__(args)

        self.global_models = [get_model(args["dataset"], args["model"]).to(args["device"])]

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

    def merge_clusters(self, clients):
        global_model_num = len(self.global_models)

        # check if there are some clusters to be merged when at least two global models exist
        loss_matrix = np.zeros((global_model_num, global_model_num))
        # generate loss matrix for calculating cluster distances
        for i in range(global_model_num):
            for j in range(global_model_num):
                total_data_size = 0
                for client in clients:
                    if client.cluster_identity == j:
                        data = client.train_set
                        total_data_size += len(data)
                        loss = get_loss(self.global_models[i], data, client.criterion, self.args["device"])
                        loss_matrix[i][j] += loss * len(data)
                # the loss is averaged across the clients in each cluster
                if total_data_size != 0:
                    loss_matrix[i][j] /= total_data_size
                else:
                    # if there is no client in cluster j, the loss is -1
                    loss_matrix[i][j] = -1

        # calculate cluster distances
        cluster_distances = np.zeros((global_model_num, global_model_num))
        for i in range(global_model_num):
            for j in range(i, global_model_num):
                if loss_matrix[i][j] == -1 or loss_matrix[j][i] == -1:
                    # there is no client in cluster i or cluster j
                    dist = -1
                else:
                    dist = max(loss_matrix[i][j] - loss_matrix[i][i], loss_matrix[j][i] - loss_matrix[j][j], 0)
                cluster_distances[i][j] = dist
                cluster_distances[j][i] = dist

        # check if there are some clusters to be merged
        deleted_models = []
        while True:
            cluster_data_size = np.zeros(global_model_num)  # number of all samples in each cluster
            for client in clients:
                if client.cluster_identity is not None:
                    cluster_data_size[client.cluster_identity] += len(client.train_set)
            cluster_i = 0
            cluster_j = 0
            min_distance = self.args["detection_threshold"]
            for i in range(global_model_num):
                for j in range(i + 1, global_model_num):
                    if cluster_distances[i][j] == -1:
                        continue
                    if cluster_distances[i][j] < min_distance:
                        cluster_i = i
                        cluster_j = j
                        min_distance = cluster_distances[i][j]

            if min_distance == self.args["detection_threshold"]:
                break

            # merge clusters
            size_i = cluster_data_size[cluster_i]
            size_j = cluster_data_size[cluster_j]
            model_i_params = parameters_to_vector(self.global_models[cluster_i].parameters())
            model_j_params = parameters_to_vector(self.global_models[cluster_j].parameters())
            merged_model_params = (size_i * model_i_params + size_j * model_j_params) / (size_i + size_j)
            print(f"\033[34mMerge cluster {cluster_i} and cluster {cluster_j}\033[0m")

            # make model i as the new model (i.e., model k in the paper)
            vector_to_parameters(merged_model_params, self.global_models[cluster_i].parameters())
            deleted_models.append(cluster_j)
            for client in clients:
                if client.cluster_identity == cluster_j:
                    client.cluster_identity = cluster_i

            for l in range(global_model_num):
                if l == cluster_i or l == cluster_j:
                    continue
                dist = max(cluster_distances[cluster_i][l], cluster_distances[cluster_j][l])
                cluster_distances[cluster_i][l] = dist
                cluster_distances[l][cluster_i] = dist

            # reset distances
            cluster_distances[:, cluster_j] = -1
            cluster_distances[cluster_j, :] = -1

        deleted_models.sort(reverse=True)
        for i in deleted_models:
            for client in clients:
                if client.cluster_identity is not None and client.cluster_identity > i:
                    client.cluster_identity -= 1
            del self.global_models[i]
