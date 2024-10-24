import torch
from copy import deepcopy
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from entities.base import Client, Server


class SCAFFOLDOptimizer(Optimizer):
    def __init__(self, params, lr, momentum=0.0, weight_decay=0.0):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        super(SCAFFOLDOptimizer, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, global_c, client_c):
        for group in self.param_groups:
            for p, c, ci in zip(group["params"], global_c, client_c):
                if p.grad is None:
                    continue

                if group['weight_decay'] != 0:
                    p.grad.data.add_(p.data, alpha=group['weight_decay'])

                grad = p.grad.data - ci + c

                if group['momentum'] != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(grad).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(group['momentum']).add_(grad, alpha=1)
                    grad = buf

                p.data.add_(grad, alpha=-group["lr"])


class SCAFFOLDClient(Client):
    def __init__(self, client_id, args, train_set, test_set, global_test_id):
        super().__init__(client_id, args, train_set, test_set, global_test_id)

        self.local_params = []
        self.local_c = [torch.zeros_like(param).to(args["device"]) for param in self.model.parameters()]
        self.local_c_update = []

    def train(self, global_c):
        """
        Perform local training with control variates.
        :param global_c: Global control variate.
        """
        self.model.train()

        train_loader = DataLoader(self.train_set, batch_size=self.args["batch_size"], shuffle=True)
        init_params = parameters_to_vector(self.model.parameters())
        global_params_list = deepcopy(list(self.model.parameters()))

        optimizer = SCAFFOLDOptimizer(self.model.parameters(), lr=self.args["lr"],
                                      weight_decay=self.args["weight_decay"], momentum=self.args["momentum"])

        for epoch in range(self.args["epochs"]):
            for _, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(device=self.args["device"]), labels.to(device=self.args["device"])

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step(global_c, self.local_c)

        with torch.no_grad():
            # update client control variate
            K = len(train_loader) * self.args["epochs"]
            self.local_c_update = []
            for c, ci, x, yi in zip(global_c, self.local_c, global_params_list, self.model.parameters()):
                ci.data = ci - c + 1 / (K * self.args["lr"]) * (x - yi)
                self.local_c_update.append(-c + 1 / (K * self.args["lr"]) * (x - yi))

            self.local_update = parameters_to_vector(self.model.parameters()) - init_params


class ScaffoldServer(Server):
    def __init__(self, args):
        super().__init__(args)

        self.global_c = [torch.zeros_like(param).to(args["device"]) for param in self.model.parameters()]

    def aggregate_by_updates(self, clients):
        update_sum = torch.zeros_like(parameters_to_vector(self.model.parameters()))
        for client in clients:
            update_sum += client.local_update

        aggregated_update = update_sum / len(clients)

        new_params = parameters_to_vector(self.model.parameters()) + self.args["global_lr"] * aggregated_update

        vector_to_parameters(new_params, self.model.parameters())

    def update_global_c(self, clients):
        for client in clients:
            for c, c_i_update in zip(self.global_c, client.local_c_update):
                c.data += c_i_update.data.clone() / self.args["client_num"]
