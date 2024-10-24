import torch
import numpy as np

from collections import Counter
from torch.optim import SGD
from torch.nn import MSELoss, CrossEntropyLoss, CosineSimilarity
from torch.utils.data import DataLoader
from entities.base import Client, Server


class FedFMClient(Client):
    def __init__(self, client_id, args, train_set, test_set, global_test_id):
        super().__init__(client_id, args, train_set, test_set, global_test_id)

        if args["penalize"] == "L2":
            self.anchor_criterion = MSELoss().to(self.args["device"])
        else:
            self.anchor_criterion = CrossEntropyLoss().to(args["device"])

        self.local_anchors = None
        self.label_distribution = torch.zeros(args["num_classes"], dtype=torch.int)

    def update_label_distribution(self):
        distribution = Counter(self.train_set.targets)
        for label, label_size in distribution.items():
            self.label_distribution[label] = label_size
        self.label_distribution = np.array(self.label_distribution)

    def train_with_anchors(self, global_anchors):
        """
        Perform local training with global anchors.
        :param global_anchors: The global anchors.
        """
        self.model.train()

        train_loader = DataLoader(self.train_set, batch_size=self.args["batch_size"], shuffle=True)

        # -------------------Start the training of representation-------------------
        optimizer = SGD(self.model.parameters(), lr=self.args["lr"], weight_decay=self.args["weight_decay"],
                        momentum=self.args["momentum"])
        cos = CosineSimilarity(dim=2).to(self.args["device"])

        for epoch in range(self.args["epochs"]):
            for _, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(device=self.args["device"]), labels.to(device=self.args["device"])

                outputs, features = self.model(inputs, True)
                loss_sup = self.criterion(outputs, labels)

                loss_anchor = 0.0
                if len(global_anchors) > 0:
                    # the global anchors is empty at the first round
                    if self.args["penalize"] == "L2":
                        # l2 alignment
                        anchors = features.clone().detach()
                        for i in range(len(labels)):
                            label = labels[i].item()
                            anchors[i] = global_anchors[label].detach()
                        loss_anchor = self.anchor_criterion(features, anchors)
                    else:
                        # contrastive alignment
                        features = features.unsqueeze(1)
                        batch_global_anchors = torch.stack(global_anchors).repeat(len(labels), 1, 1)
                        logits = cos(features, batch_global_anchors)
                        loss_anchor = self.anchor_criterion(logits / self.args["temperature"], labels)

                loss = loss_sup + self.args["lambda"] * loss_anchor

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        with torch.no_grad():
            # update local prototypes after local training
            self.local_anchors = self.get_local_anchors(self.model)

    def get_local_anchors(self, model):
        """
        Get the local anchors for each label.
        :param model: The local model.
        :return: The average of anchors for each label.
        """
        anchor_dict = {}
        train_loader = DataLoader(self.train_set, batch_size=2048, shuffle=True)

        with torch.no_grad():
            for _, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(device=self.args["device"]), labels.to(device=self.args["device"])
                outputs, features = model(inputs, True)
                anchors = features.clone().detach()
                for i in range(len(labels)):
                    label = labels[i].item()
                    if label in anchor_dict:
                        anchor_dict[label].append(anchors[i, :])
                    else:
                        anchor_dict[label] = [anchors[i, :]]

        # get the average of prototypes for each class
        anchor_mean = {}
        for label, anchor_list in anchor_dict.items():
            anchor_mean[label] = torch.mean(torch.stack(anchor_list), dim=0)

        return anchor_mean


class FedFMServer(Server):
    def __init__(self, args):
        super().__init__(args)

        self.global_anchors = []

    def aggregate_anchors(self, clients):
        """
        Aggregate the global anchors according to the label distributions of clients.
        """
        aggregate_proto_dict = {}
        label_size_dict = {}

        for client in clients:
            # get this client's label distribution
            label_distribution = client.label_distribution
            local_anchors = client.local_anchors
            for label in local_anchors.keys():
                if label in aggregate_proto_dict:
                    aggregate_proto_dict[label] += local_anchors[label] * label_distribution[label]
                    label_size_dict[label] += label_distribution[label]
                else:
                    aggregate_proto_dict[label] = local_anchors[label] * label_distribution[label]
                    label_size_dict[label] = label_distribution[label]

        for label, proto in aggregate_proto_dict.items():
            aggregate_proto_dict[label] = proto / label_size_dict[label]

        self.global_anchors = [aggregate_proto_dict[label] for label in range(self.args["num_classes"])]
