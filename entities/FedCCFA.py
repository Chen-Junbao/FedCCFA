import torch
import numpy as np

from collections import Counter
from typing import Iterator
from torch.optim import SGD
from torch.nn import Parameter, MSELoss, CosineSimilarity, CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from sklearn.cluster import DBSCAN
from entities.base import Client, Server
from utils.gen_dataset import ClientDataset


class FedCCFAClient(Client):
    def __init__(self, client_id, args, train_set, test_set, global_test_id):
        super().__init__(client_id, args, train_set, test_set, global_test_id)

        self.clf_keys = []
        if args["penalize"] == "L2":
            self.proto_criterion = MSELoss().to(self.args["device"])
        else:
            self.proto_criterion = CrossEntropyLoss().to(args["device"])

        self.local_protos = {}
        self.global_protos = []
        self.p_clf_params = []
        self.label_distribution = torch.zeros(args["num_classes"], dtype=torch.int)
        self.proto_weight = 0.0

    def update_label_distribution(self):
        distribution = Counter(self.train_set.targets)
        for label, label_size in distribution.items():
            self.label_distribution[label] = label_size
        self.label_distribution = np.array(self.label_distribution)
        prob = self.label_distribution / len(self.train_set.targets)
        entropy = -np.sum(prob * np.log(prob))
        if self.args["gamma"] != 0:
            # adaptively adjust the weight of prototype loss according to the entropy of the training set
            self.proto_weight = entropy / self.args["gamma"]
        else:
            self.proto_weight = self.args["lambda"]

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

    def set_label_params(self, label, new_params: Iterator[Parameter]):
        """
        Set local model's classifier parameters as new parameters.
        :param label: The label to be set.
        :param new_params: The new parameters to be set.
        """
        clf_params = [param for name, param in self.model.named_parameters() if name in self.clf_keys]
        for new_param, local_param in zip(new_params, clf_params):
            local_param.data[label] = new_param.data.clone()

    def get_clf_parameters(self):
        """
        Return local model's classifier parameters.
        :return: The parameters of local classifier.
        """
        clf_params = [param for name, param in self.model.named_parameters() if name in self.clf_keys]

        return clf_params

    def train_with_protos(self, _round):
        """
        Perform decoupled local training.
        First, train the classifier for clf_epoch. Then, train the representation for rep_epoch.
        """
        self.set_clf_params(self.p_clf_params)
        self.model.train()

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
        cos = CosineSimilarity(dim=2).to(self.args["device"])

        for epoch in range(self.args["rep_epochs"]):
            for _, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(device=self.args["device"]), labels.to(device=self.args["device"])

                outputs, features = self.model(inputs, True)
                loss_sup = self.criterion(outputs, labels)

                loss_proto = 0.0
                if len(self.global_protos) > 0 and _round >= 20:
                    # the global prototypes is empty at the first round
                    if self.proto_weight != 0:
                        if self.args["penalize"] == "L2":
                            # l2 alignment
                            protos = features.clone().detach()
                            for i in range(len(labels)):
                                label = labels[i].item()
                                protos[i] = self.global_protos[label].detach()
                            loss_proto = self.proto_criterion(features, protos)
                        else:
                            # contrastive alignment
                            features = features.unsqueeze(1)
                            batch_global_protos = torch.stack(self.global_protos).repeat(len(labels), 1, 1)
                            logits = cos(features, batch_global_protos)
                            loss_proto = self.proto_criterion(logits / self.args["temperature"], labels)

                loss = loss_sup + self.proto_weight * loss_proto

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        with torch.no_grad():
            # update local prototypes after local training
            self.local_protos = self.get_local_protos(self.model)
            self.global_protos = [self.local_protos[label] for label in range(self.args["num_classes"])]

    def balance_train(self):
        """
        Train classifiers under balanced training set.
        """
        self.model.train()

        # -------------------Start the training of classifier-------------------
        balanced_train_set = self.class_balance_sample()
        train_loader = DataLoader(balanced_train_set, batch_size=self.args["batch_size"], shuffle=True)
        for name, param in self.model.named_parameters():
            if name in self.clf_keys:
                param.requires_grad = True
            else:
                param.requires_grad = False

        optimizer = SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.args["balanced_clf_lr"],
                        weight_decay=self.args["weight_decay"], momentum=self.args["momentum"])

        for epoch in range(self.args["balanced_epochs"]):
            for _, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(device=self.args["device"]), labels.to(device=self.args["device"])

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def class_balance_sample(self):
        indices = []
        labels = np.array(self.train_set.targets)
        num_per_class = Counter(labels)
        min_size = num_per_class.most_common()[-1][1]
        min_size = min(min_size, 5)

        classes = set(labels)
        for i in range(len(classes)):
            class_indices = np.where(labels == i)[0]
            indices.extend(np.random.choice(class_indices, min_size, replace=False))
        np.random.shuffle(indices)

        return ClientDataset(self.train_set, indices, self.train_set.transform)

    def get_local_protos(self, model):
        """
        Get the local prototypes for each class.
        :param model: The local model.
        :return: The average of prototypes for each class.
        """
        proto_dict = {}
        train_loader = DataLoader(self.train_set, batch_size=2048, shuffle=True)

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

    def fine_tune(self):
        """
        Fine-tune the classifier.
        """
        self.model.train()

        train_loader = DataLoader(self.train_set, batch_size=self.args["batch_size"], shuffle=True)

        # -------------------Start the training of classifier-------------------
        for name, param in self.model.named_parameters():
            if name in self.clf_keys:
                param.requires_grad = True
            else:
                param.requires_grad = False

        optimizer = SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.args["p_lr"],
                        weight_decay=self.args["weight_decay"], momentum=self.args["momentum"])

        for epoch in range(5):
            for _, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(device=self.args["device"]), labels.to(device=self.args["device"])

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


class FedCCFAServer(Server):
    def __init__(self, args):
        super().__init__(args)

        self.clf_keys = None
        self.global_protos = []
        self.prev_rep_norm = 0
        self.rep_norm_scale = 0
        self.prev_clf_norm = 0
        self.clf_norm_scale = 0

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
        if self.args["weights"] == "uniform":
            aggregate_proto_dict = {}

            for client in clients:
                local_protos = client.local_protos
                for label in local_protos.keys():
                    if label in aggregate_proto_dict:
                        aggregate_proto_dict[label] += local_protos[label]
                    else:
                        aggregate_proto_dict[label] = local_protos[label]

            for label, proto in aggregate_proto_dict.items():
                aggregate_proto_dict[label] = proto / len(clients)
        else:
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

        self.global_protos = [aggregate_proto_dict[label] for label in range(self.args["num_classes"])]

    @staticmethod
    def madd(vecs):
        def cos_sim(a, b):
            return a.dot(b) / (np.linalg.norm(a) * np.linalg.norm(b))

        num = len(vecs)
        res = np.zeros((num, num))
        for i in range(num):
            for j in range(i + 1, num):
                dist = 0.0
                for z in range(num):
                    if z == i or z == j:
                        continue

                    dist += np.abs(cos_sim(vecs[i], vecs[z]) - cos_sim(vecs[j], vecs[z]))
                res[i][j] = res[j][i] = dist / (num - 2)

        return res

    def merge_classifiers(self, clf_params_dict):
        """
        Merge the classifiers under the same concepts by measuring the parameter distance of each label.
        :param clf_params_dict: The classifier parameters of all clients. Key is client id, and value is its parameters.
        :return: The merged client ids for each label. Key is the label, and value is the list of client clusters.
        """
        client_ids = np.array(list(clf_params_dict.keys()))
        client_clf_params = list(clf_params_dict.values())
        label_num = self.args["num_classes"]

        label_merged_dict = {}
        for label in range(label_num):
            # calculate the distance matrix of each label by the parameters of classifiers
            params_list = []
            for clf_params in client_clf_params:
                params = [param[label].detach().cpu().numpy() for param in clf_params]
                params_list.append(np.hstack(params))
            params_list = np.array(params_list)

            dist = self.madd(params_list)
            np.set_printoptions(threshold=np.inf, linewidth=np.inf)
            # print(label, np.round(dist, 2))

            clustering = DBSCAN(eps=self.args["eps"], min_samples=1, metric="precomputed")
            clustering.fit(dist)

            merged_ids = []
            for i in set(clustering.labels_):
                indices = np.where(clustering.labels_ == i)[0]
                if len(indices) > 1:
                    # there are at least two classifiers to be merged for this label
                    ids = client_ids[indices]
                    ids.sort()  # sort for better observation
                    merged_ids.append(list(ids))

            label_merged_dict[label] = merged_ids

        return label_merged_dict

    def oracle_merging(self, _round, ids):
        if _round < 100:
            return {
                0: [ids],
                1: [ids],
                2: [ids],
                3: [ids],
                4: [ids],
                5: [ids],
                6: [ids],
                7: [ids],
                8: [ids],
                9: [ids]
            }
        else:
            return {
                0: [ids],
                1: [[_id for _id in ids if 0 <= _id % 10 < 3], [_id for _id in ids if _id % 10 >= 3]],
                2: [[_id for _id in ids if 0 <= _id % 10 < 3], [_id for _id in ids if _id % 10 >= 3]],
                3: [[_id for _id in ids if 3 <= _id % 10 < 6], [_id for _id in ids if not 3 <= _id % 10 < 6]],
                4: [[_id for _id in ids if 3 <= _id % 10 < 6], [_id for _id in ids if not 3 <= _id % 10 < 6]],
                5: [[_id for _id in ids if _id % 10 >= 6], [_id for _id in ids if _id % 10 < 6]],
                6: [[_id for _id in ids if _id % 10 >= 6], [_id for _id in ids if _id % 10 < 6]],
                7: [ids],
                8: [ids],
                9: [ids]
            }

    def aggregate_label_params(self, label, clients):
        """
        Aggregate the label parameters.
        :param label: The label to be aggregated.
        :param clients: Clients within the same group.
        """
        label_params = [param[label] for name, param in self.model.named_parameters() if name in self.clf_keys]
        label_size = 0
        aggregated_params = torch.zeros_like(parameters_to_vector(label_params))

        for client in clients:
            client_label_params = [param[label] for name, param in client.model.named_parameters()
                                   if name in self.clf_keys]
            client_params = parameters_to_vector(client_label_params)
            if self.args["weights"] == "uniform":
                aggregated_params += client_params
            else:
                aggregated_params += client_params * client.label_distribution[label]
                label_size += client.label_distribution[label]

        if self.args["weights"] == "uniform":
            aggregated_params /= len(clients)
        else:
            aggregated_params /= label_size
        aggregated_params.to(self.args["device"])

        return aggregated_params

    def aggregate_label_protos(self, label, clients):
        """
        Aggregate the label prototypes.
        :param label: The label to be aggregated.
        :param clients: Clients within the same group.
        """
        if self.args["weights"] == "uniform":
            aggregated_proto = None

            for client in clients:
                if aggregated_proto is None:
                    aggregated_proto = client.local_protos[label]
                else:
                    aggregated_proto += client.local_protos[label]

            aggregated_proto /= len(clients)
        else:
            aggregated_proto = None
            label_size = 0

            for client in clients:
                if aggregated_proto is None:
                    aggregated_proto = client.local_protos[label] * client.label_distribution[label]
                    label_size = client.label_distribution[label]
                else:
                    aggregated_proto += client.local_protos[label] * client.label_distribution[label]
                    label_size += client.label_distribution[label]

            aggregated_proto /= label_size

        return aggregated_proto
