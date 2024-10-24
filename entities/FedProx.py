import torch

from torch.optim import Optimizer
from torch.utils.data import DataLoader
from entities.base import Client, Server


class PerturbedGradientDescent(Optimizer):
    def __init__(self, params, lr, mu=0.0, momentum=0.0, weight_decay=0.0):
        defaults = dict(lr=lr, mu=mu, momentum=momentum, weight_decay=weight_decay)
        super(PerturbedGradientDescent, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, global_params, device):
        for group in self.param_groups:
            for p, g in zip(group['params'], global_params):
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if group['weight_decay'] != 0:
                    d_p.add_(p.data, alpha=group['weight_decay'])
                if group['momentum'] != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(group['momentum']).add_(d_p)
                    d_p = buf
                g = g.to(device)
                d_p.add_(group['mu'] * (p.data - g.data))
                p.data.add_(d_p, alpha=-group['lr'])


class FedProxClient(Client):
    def __init__(self, client_id, args, train_set, test_set, global_test_id):
        super().__init__(client_id, args, train_set, test_set, global_test_id)

    def train(self):
        """
        Perform local training using mini-batch SGD.
        """
        global_params = [param.data.clone() for param in self.model.parameters()]

        self.model.train()

        train_loader = DataLoader(self.train_set, batch_size=self.args["batch_size"], shuffle=True)

        optimizer = PerturbedGradientDescent(self.model.parameters(), lr=self.args["lr"], mu=self.args["mu"],
                                             weight_decay=self.args["weight_decay"], momentum=self.args["momentum"])

        for epoch in range(self.args["epochs"]):
            for _, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(device=self.args["device"]), labels.to(device=self.args["device"])

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step(global_params, self.args["device"])


class FedProxServer(Server):
    def __init__(self, args):
        super().__init__(args)
