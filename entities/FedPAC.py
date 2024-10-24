import torch
import numpy as np
import cvxpy as cvx

from typing import Iterator
from collections import Counter
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.nn import MSELoss, Parameter
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from entities.base import Client, Server


class FedPACClient(Client):
    def __init__(self, client_id, args, train_set, test_set, global_test_id):
        super().__init__(client_id, args, train_set, test_set, global_test_id)

        self.mse_criterion = MSELoss().to(args["device"])
        self.clf_keys = None
        self.label_distribution = torch.zeros(args["num_classes"])
        self.label_prob = torch.zeros(args["num_classes"])
        self.local_protos = None

    def set_rep_params(self, new_params: Iterator[Parameter]):
        """
        Set local model's representation parameters as new parameters.
        :param new_params: The new parameters to be set.
        """
        rep_params = [param for name, param in self.model.named_parameters() if name not in self.clf_keys]
        for new_param, local_param in zip(new_params, rep_params):
            local_param.data = new_param.data.clone()

    def set_clf_params(self, new_params: Iterator[Parameter]):
        """
        Set local model's classifier parameters as new parameters.
        :param new_params: The new parameters to be set.
        """
        clf_params = [param for name, param in self.model.named_parameters() if name in self.clf_keys]
        for new_param, local_param in zip(new_params, clf_params):
            local_param.data = new_param.data.clone()

    def train_with_protos(self, global_protos):
        """
        Perform decoupled local training with global prototypes.
        First, train the classifier for clf_epoch. Then, train the representation for rep_epoch.
        :param global_protos: The global prototypes.
        """
        self.model.train()

        # get local prototypes before local training
        local_protos = self.get_local_protos(self.model)

        train_loader = DataLoader(self.train_set, batch_size=self.args["batch_size"], shuffle=True)

        # -------------------Start the training of classifier-------------------
        for name, param in self.model.named_parameters():
            if name in self.clf_keys:
                param.requires_grad = True
            else:
                param.requires_grad = False

        optimizer = SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.args["clf_lr"],
                        weight_decay=self.args["weight_decay"], momentum=self.args["momentum"])

        for epoch in range(self.args["clf_epochs"]):
            for _, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(device=self.args["device"]), labels.to(device=self.args["device"])

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # -------------------Start the training of representation-------------------
        for name, param in self.model.named_parameters():
            if name in self.clf_keys:
                param.requires_grad = False
            else:
                param.requires_grad = True

        optimizer = SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.args["rep_lr"],
                        weight_decay=self.args["weight_decay"], momentum=self.args["momentum"])

        for epoch in range(self.args["rep_epochs"]):
            for _, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(device=self.args["device"]), labels.to(device=self.args["device"])

                outputs, features = self.model(inputs, True)
                loss_sup = self.criterion(outputs, labels)
                loss_proto = 0.0

                if len(global_protos) > 0:
                    # the global prototypes is empty at the first round
                    protos = features.clone().detach()
                    for i in range(len(labels)):
                        label = labels[i].item()
                        if label in global_protos:
                            protos[i] = global_protos[label].detach()
                        else:
                            protos[i] = local_protos[label].detach()
                    loss_proto = self.mse_criterion(features, protos)
                loss = loss_sup + self.args["lambda"] * loss_proto

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        with torch.no_grad():
            # update local prototypes after local training
            self.local_protos = self.get_local_protos(self.model)

    def get_local_protos(self, model):
        """
        Get the local prototypes for each class.
        :param model: The local model.
        :return: The average of prototypes for each class.
        """
        proto_dict = {}
        train_loader = DataLoader(self.train_set, batch_size=self.args["batch_size"], shuffle=True)

        with torch.no_grad():
            for _, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(device=self.args["device"]), labels.to(device=self.args["device"])
                outputs, features = model(inputs, True)
                protos = features.clone().detach()
                for i in range(len(labels)):
                    label = labels[i].item()
                    if label in proto_dict:
                        proto_dict[label].append(protos[i, :])
                    else:
                        proto_dict[label] = [protos[i, :]]

        # get the average of prototypes for each class
        proto_mean = {}
        for label, proto_list in proto_dict.items():
            proto_mean[label] = torch.mean(torch.stack(proto_list), dim=0)

        return proto_mean

    def statistics_extraction(self):
        g_params = self.model.state_dict()[self.clf_keys[0]]
        d = g_params.shape[1]
        feature_dict = {}
        train_loader = DataLoader(self.train_set, batch_size=self.args["batch_size"], shuffle=True)

        with torch.no_grad():
            for _, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(device=self.args["device"]), labels.to(device=self.args["device"])
                outputs, features = self.model(inputs, True)
                protos = features.clone().detach().cpu()

                for i in range(len(labels)):
                    label = labels[i].item()
                    if label in feature_dict:
                        feature_dict[label].append(protos[i, :])
                    else:
                        feature_dict[label] = [protos[i, :]]
        for label, feature_list in feature_dict.items():
            feature_dict[label] = torch.stack(feature_list)

        py = self.label_prob
        py2 = py.mul(py)
        v = 0
        h_ref = torch.zeros((self.args["num_classes"], d))

        for label in range(self.args["num_classes"]):
            if label in feature_dict:
                label_feature = feature_dict[label]
                label_num = label_feature.shape[0]
                label_feature_mu = label_feature.mean(dim=0)
                h_ref[label] = py[label] * label_feature_mu
                v += (py[label] * torch.trace((torch.mm(torch.t(label_feature), label_feature) / label_num))).item()
                v -= (py2[label] * (torch.mul(label_feature_mu, label_feature_mu))).sum().item()

        v /= len(self.train_set)

        return v, h_ref

    def update_label_distribution(self):
        distribution = Counter(self.train_set.targets)
        for label, label_size in distribution.items():
            self.label_distribution[label] = label_size
            self.label_prob[label] = label_size / len(self.train_set)


class FedPACServer(Server):
    def __init__(self, args):
        super().__init__(args)

        self.clf_keys = None
        self.global_protos = {}

    def send_rep_params(self, clients):
        rep_params = [param for name, param in self.model.named_parameters() if name not in self.clf_keys]
        for client in clients:
            client.set_rep_params(rep_params)

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

    def aggregate_protos(self, clients):
        """
        Aggregate the global prototypes according to the label distributions of clients.
        """
        aggregate_proto_dict = {}
        label_size_dict = {}

        for client in clients:
            # get this client's label distribution
            label_distribution = client.label_distribution
            local_protos = client.local_protos
            for label in local_protos.keys():
                if label in aggregate_proto_dict:
                    aggregate_proto_dict[label] += local_protos[label] * label_distribution[label]
                    label_size_dict[label] += label_distribution[label]
                else:
                    aggregate_proto_dict[label] = local_protos[label] * label_distribution[label]
                    label_size_dict[label] = label_distribution[label]

        for label, proto in aggregate_proto_dict.items():
            aggregate_proto_dict[label] = proto / label_size_dict[label]

        self.global_protos = aggregate_proto_dict

    def get_head_agg_weight(self, Vars, Hs, client_num):
        def pairwise(data):
            """ Simple generator of the pairs (x, y) in a tuple such that index x < index y.
            Args:
            data Indexable (including ability to query length) containing the elements
            Returns:
            Generator over the pairs of the elements of 'data'
            """
            n = len(data)
            for i in range(n):
                for j in range(i, n):
                    yield data[i], data[j]

        num_cls = Hs[0].shape[0]  # number of classes
        d = Hs[0].shape[1]  # dimension of feature representation
        v = torch.tensor(Vars)
        avg_weight = []
        for index in range(client_num):
            h_ref = Hs[index]
            dist = torch.zeros((client_num, client_num))
            for j1, j2 in pairwise(tuple(range(client_num))):
                h_j1 = Hs[j1]
                h_j2 = Hs[j2]
                h = torch.zeros((d, d))
                for k in range(num_cls):
                    h += torch.mm((h_ref[k] - h_j1[k]).reshape(d, 1), (h_ref[k] - h_j2[k]).reshape(1, d))
                dj12 = torch.trace(h)
                dist[j1][j2] = dj12
                dist[j2][j1] = dj12

            # QP solver
            p_matrix = torch.diag(v) + dist
            p_matrix = p_matrix.numpy()  # coefficient for QP problem
            eigenvalues, eigenvectors = torch.linalg.eig(torch.tensor(p_matrix))

            # for numerical stability
            p_matrix_new = 0
            for ii in range(client_num):
                real_value = torch.view_as_real(eigenvalues[ii])[0]
                if real_value >= 0.01:
                    p_matrix_new += real_value * torch.mm(eigenvectors[:, ii].reshape(client_num, 1),
                                                          eigenvectors[:, ii].reshape(1, client_num))
            p_matrix = p_matrix_new.numpy() if not np.all(np.linalg.eigvals(p_matrix) >= 0.0) else p_matrix

            # solve QP
            eps = 1e-3
            if np.all(np.linalg.eigvals(p_matrix) >= 0):
                alphav = cvx.Variable(client_num)
                obj = cvx.Minimize(cvx.quad_form(alphav, p_matrix))
                prob = cvx.Problem(obj, [cvx.sum(alphav) == 1.0, alphav >= 0])
                prob.solve()
                alpha = alphav.value
                alpha = [i * (i > eps) for i in alpha]  # zero-out small weights (<eps)
                # if i == 0:
                #     print('({}) Agg Weights of Classifier Head'.format(i + 1))
                #     print(alpha, '\n')

            else:
                alpha = None  # if no solution for the optimization problem, use local classifier only

            avg_weight.append(alpha)

        return avg_weight

    def aggregate_clf(self, all_clf_params, avg_weight):
        new_params = torch.zeros_like(all_clf_params[0])
        for i in range(len(all_clf_params)):
            new_params += avg_weight[i] * all_clf_params[i]

        return new_params
