import numpy as np


def drift_dataset(dataset, class_a, class_b):
    """
    Perform concept drift by swapping two labels class_a and class_b.
    :param dataset: The dataset to be drifted.
    :param class_a: The drifted label.
    :param class_b: The drifted label.
    :return:
    """
    class_a_indices = np.where(np.array(dataset.targets) == class_a)[0]
    class_b_indices = np.where(np.array(dataset.targets) == class_b)[0]
    for i in class_a_indices:
        dataset.targets[i] = -1
    for i in class_b_indices:
        dataset.targets[i] = class_a
    class_a_indices = np.where(np.array(dataset.targets) == -1)[0]
    for i in class_a_indices:
        dataset.targets[i] = class_b


def sudden_drift(clients, global_test_sets, _round):
    print(f"Sudden drift occurs at {_round}")

    drift_dataset(global_test_sets[1], 1, 2)
    drift_dataset(global_test_sets[2], 3, 4)
    drift_dataset(global_test_sets[3], 5, 6)

    for client in clients:
        if client.id % 10 < 3:
            drift_dataset(client.train_set, 1, 2)
            drift_dataset(client.test_set, 1, 2)
            client.global_test_id = 1
        elif client.id % 10 < 6:
            drift_dataset(client.train_set, 3, 4)
            drift_dataset(client.test_set, 3, 4)
            client.global_test_id = 2
        else:
            drift_dataset(client.train_set, 5, 6)
            drift_dataset(client.test_set, 5, 6)
            client.global_test_id = 3


def incremental_drift(clients, global_test_sets, _round):
    print(f"Incremental drift occurs at {_round}")

    if _round == 100:
        drift_dataset(global_test_sets[1], 1, 2)
    elif _round == 110:
        drift_dataset(global_test_sets[2], 3, 4)
    elif _round == 120:
        drift_dataset(global_test_sets[3], 5, 6)

    for client in clients:
        if _round == 100 and client.id % 10 < 3:
            drift_dataset(client.train_set, 1, 2)
            drift_dataset(client.test_set, 1, 2)
            client.global_test_id = 1
        elif _round == 110 and 3 <= client.id % 10 < 6:
            drift_dataset(client.train_set, 3, 4)
            drift_dataset(client.test_set, 3, 4)
            client.global_test_id = 2
        elif _round == 120 and client.id % 10 >= 6:
            drift_dataset(client.train_set, 5, 6)
            drift_dataset(client.test_set, 5, 6)
            client.global_test_id = 3
