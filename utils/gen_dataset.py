import random
import numpy as np

from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import transforms


class ClientDataset(Dataset):
    def __init__(self, dataset=None, indices=None, transform=None):
        if isinstance(dataset, datasets.folder.ImageFolder):
            self.data = []
            self.targets = []
            for index in indices:
                self.data.append(dataset[index][0])
                self.targets.append(dataset[index][1])
            self.data = np.array(self.data)
            self.targets = np.array(self.targets)
        else:
            self.data = dataset.data[indices]
            self.targets = np.array(dataset.targets)[indices]
        self.transform = transform

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        sample, target = self.data[index], self.targets[index]

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, target


def partition_dataset(dataset, client_num, partition, alpha, essential_num, seed):
    """
    Partition the original dataset according to different partition strategies.
    :param dataset: The original dataset to be partitioned.
    :param client_num: The number of all clients.
    :param partition: Partition strategy (homo or hetero).
    :param alpha: The parameter of Dirichlet distribution.
    :param essential_num: Number of essential training samples to adapt to concept drift for each class.
    :param seed: Random seed.
    :return: The data indices for each client.
    """
    labels = np.array(dataset.targets)
    classes = set(labels)

    if partition == "homo":
        client_indices = [[] for _ in range(client_num)]  # dataset indices for all clients

        for k in range(len(classes)):
            class_indices = np.where(labels == k)[0]
            size = len(class_indices) // client_num  # size of data owned by each client
            for i in range(client_num):
                client_indices[i].extend(class_indices[i * size: (i + 1) * size])
    else:
        # partition dataset according to Dirichlet distribution
        client_indices = [[] for _ in range(client_num)]  # dataset indices for all clients

        for k in range(len(classes)):
            class_indices = np.where(labels == k)[0]

            # for each client's training set, essential samples are distributed for each class
            essential_samples = np.array_split(class_indices[:client_num * essential_num], client_num)
            client_indices = [idx_j + idx.tolist() for idx_j, idx in zip(client_indices, essential_samples)]
            class_indices = class_indices[client_num * essential_num:]

            # the remaining samples are distributed according to Dirichlet distribution
            random.seed(seed + k)  # "+k" makes the Dirichlet distribution different for each class
            np.random.seed(seed + k)
            proportions = np.random.dirichlet(np.repeat(alpha, client_num))
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(class_indices)).astype(int)[:-1]
            client_indices = [idx_j + idx.tolist() for idx_j, idx in
                              zip(client_indices, np.split(class_indices, proportions))]

    for i in range(client_num):
        random.shuffle(client_indices[i])

    return client_indices


def distribute_dataset(name, client_num, partition, alpha, seed):
    """
    Distribute the dataset to all clients.
    :param name: The name of dataset.
    :param client_num: The number of clients.
    :param partition: Partition strategy (homo or hetero).
    :param alpha: The parameter of Dirichlet distribution.
    :param seed: Random seed.
    :return: Dict[client_id, dataset] for training set and test set.
    """
    if name == "Fashion-MNIST":
        train_set = datasets.FashionMNIST(root="../../../data", train=True, download=True)
        train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        test_set = datasets.FashionMNIST(root="../../../data", train=False, download=True)
        test_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    elif name == "CIFAR10":
        train_set = datasets.CIFAR10(root="../../../data", train=True, download=True)
        train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop(32, 4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
        ])
        test_set = datasets.CIFAR10(root="../../../data", train=False, download=True)
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
        ])
    elif name == "CINIC-10":
        train_set = datasets.ImageFolder("../../../data/CINIC-10/train")
        train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.47889522, 0.47227842, 0.43047404), std=(0.24205776, 0.23828046, 0.25874835))
        ])
        test_set = datasets.ImageFolder("../../../data/CINIC-10/test")
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.47889522, 0.47227842, 0.43047404), std=(0.24205776, 0.23828046, 0.25874835))
        ])
    else:
        train_set, test_set = None, None
        train_transform, test_transform = None, None

    client_train_indices = partition_dataset(train_set, client_num, partition, alpha, 5, seed)
    client_test_indices = partition_dataset(test_set, client_num, partition, alpha, 2, seed)

    client_train_set, client_test_set = {}, {}
    for client_id in range(client_num):
        client_train_set[client_id] = ClientDataset(train_set, client_train_indices[client_id],
                                                    transform=train_transform)
        client_test_set[client_id] = ClientDataset(test_set, client_test_indices[client_id], transform=test_transform)

    global_test_sets = [ClientDataset(test_set, list(range(len(test_set))), transform=test_transform) for _ in range(4)]

    return client_train_set, client_test_set, global_test_sets
