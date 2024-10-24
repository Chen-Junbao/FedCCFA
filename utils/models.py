import torch.nn as nn


def get_model(dataset_name, model):
    if model == "CNN":
        if dataset_name in ["Fashion-MNIST"]:
            return FashionMNISTCNN()
        elif dataset_name in ["CIFAR10", "CINIC-10"]:
            return CifarCNN()


class FashionMNISTCNN(nn.Module):
    def __init__(self):
        super(FashionMNISTCNN, self).__init__()

        self.hidden_layers = nn.Sequential(
            nn.Conv2d(1, 16, 5, padding=0),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 5, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(32 * 5 * 5, 128),
            nn.LeakyReLU()
        )
        self.fc = nn.Linear(128, 10)

    def forward(self, x, return_features=False):
        feature = self.hidden_layers(x)
        out = self.fc(feature)

        if return_features:
            return out, feature
        else:
            return out


class CifarCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CifarCNN, self).__init__()

        self.hidden_layers = nn.Sequential(
            nn.Conv2d(3, 16, 5, padding=0),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 5, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(64 * 3 * 3, 128),
            nn.LeakyReLU()
        )
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x, return_features=False):
        feature = self.hidden_layers(x)
        out = self.fc(feature)

        if return_features:
            return out, feature
        else:
            return out
