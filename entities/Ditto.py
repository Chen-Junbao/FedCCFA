import torch

from copy import deepcopy
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from entities.base import Client, Server
from utils.metric import get_accuracy


class DittoOptimizer(Optimizer):
    def __init__(self, params, lr, _lambda=0.0, momentum=0.0, weight_decay=0.0):
        defaults = dict(lr=lr, _lambda=_lambda, momentum=momentum, weight_decay=weight_decay)
        super(DittoOptimizer, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, global_params):
        for group in self.param_groups:
            for p, g in zip(group['params'], global_params):
                if p.grad is None:
                    continue

                if group['weight_decay'] != 0:
                    p.grad.data.add_(p.data, alpha=group['weight_decay'])

                grad = p.grad.data + group['_lambda'] * (p.data - g.data)

                if group['momentum'] != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(grad).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(group['momentum']).add_(grad)
                    grad = buf

                p.data.add_(grad, alpha=-group['lr'])


class DittoClient(Client):
    def __init__(self, client_id, args, train_set, test_set, global_test_id):
        super().__init__(client_id, args, train_set, test_set, global_test_id)

        self.p_model = deepcopy(self.model)  # personalized model

    def personalized_train(self, global_model):
        global_params = [param.data.clone() for param in global_model.parameters()]

        self.p_model.train()
        train_loader = DataLoader(self.train_set, batch_size=self.args["batch_size"], shuffle=True)
        optimizer = DittoOptimizer(self.p_model.parameters(), lr=self.args["lr"], _lambda=self.args["lambda"],
                                   weight_decay=self.args["weight_decay"], momentum=self.args["momentum"])

        for epoch in range(self.args["local_epochs"]):
            for _, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(device=self.args["device"]), labels.to(device=self.args["device"])

                outputs = self.p_model(inputs)
                loss = self.criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step(global_params)

    def local_test(self):
        local_accuracy = get_accuracy(self.p_model, self.test_set, self.args["device"])

        return local_accuracy

    def global_test(self, global_tests):
        global_accuracy = get_accuracy(self.p_model, global_tests[self.global_test_id], self.args["device"])

        return global_accuracy


class DittoServer(Server):
    def __init__(self, args):
        super().__init__(args)
