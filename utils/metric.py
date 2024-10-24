import torch
import numpy as np
import pandas as pd

from datetime import datetime
from torch.utils.data import DataLoader


def get_accuracy(model, dataset, device):
    data_loader = DataLoader(dataset, batch_size=4096)
    correct, total = 0, 0
    with torch.no_grad():
        model.eval()

        for _, (inputs, labels) in enumerate(data_loader):
            inputs, labels = inputs.to(device=device), labels.to(device=device)

            outputs = model(inputs)
            _, pred = torch.max(outputs.data, 1)

            total += inputs.data.size()[0]
            correct += (pred == labels.data).sum().item()

        accuracy = correct / float(total)

    return accuracy


def get_loss(model, dataset, criterion, device):
    data_loader = DataLoader(dataset, batch_size=4096)
    with torch.no_grad():
        model.eval()

        total_loss = 0.0
        true_labels = np.array(dataset.targets)

        for _, (inputs, labels) in enumerate(data_loader):
            inputs, labels = inputs.to(device=device), labels.to(device=device)

            outputs = model(inputs)
            avg_batch_loss = criterion(outputs, labels)
            total_loss += avg_batch_loss.item() * outputs.shape[0]

        avg_loss = total_loss / len(true_labels)

    return avg_loss


def save_results(info, args):
    logger = pd.DataFrame(columns=list(info[0].keys()), data=info)
    cur_time = datetime.now().strftime("%Y%m%d%H%M%S%f")
    logger.to_csv(f"../../log/{args['dataset']}_{args['drift_pattern']}_{args['algorithm']}_{cur_time}.csv",
                  index=False)

    print("accuracy summary:")
    print(logger["local_accuracy"].mean())
    print(logger["global_accuracy"].mean())
