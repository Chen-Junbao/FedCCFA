import yaml
import json

from copy import deepcopy
from entities.FedDrift import FedDriftClient, FedDriftServer
from utils.models import get_model
from utils.drift import sudden_drift, incremental_drift
from utils.gen_dataset import distribute_dataset

if __name__ == "__main__":
    # load configuration
    with open("../configs/FedDrift.yaml", 'r') as f:
        args = yaml.load(f, Loader=yaml.FullLoader)
        print(json.dumps(args, indent=4))

    # initialize clients, server and global model
    client_train_set, client_test_set, global_test_sets = distribute_dataset(
        args["dataset"], args["client_num"], args["partition"], args["alpha"], args["seed"]
    )

    clients = []
    for client_id in range(args["client_num"]):
        client = FedDriftClient(client_id, args, client_train_set[client_id], client_test_set[client_id], 0)
        clients.append(client)

    server = FedDriftServer(args)
    server.get_client_data_size(clients)

    for _round in range(args["rounds"]):
        if args["drift_pattern"] == "sudden" and _round == 100:
            sudden_drift(clients, global_test_sets, _round)
        elif args["drift_pattern"] == "recurrent" and _round in [100, 150]:
            sudden_drift(clients, global_test_sets, _round)
        elif args["drift_pattern"] == "incremental" and _round in [100, 110, 120]:
            incremental_drift(clients, global_test_sets, _round)

        selected_clients = server.select_clients(clients)

        for client in selected_clients:
            client.clustering(server.global_models)

        if len(server.global_models) > 1:
            server.merge_clusters(selected_clients)

        for client in selected_clients:
            if client.cluster_identity is None:
                # for every drifted client, initialize a model to be added to the set of global models at the next round
                server.global_models.append(get_model(args["dataset"], args["model"]).to(args["device"]))
                client.cluster_identity = len(server.global_models) - 1

        # count the number of used global models
        cluster_identities = [client.cluster_identity for client in clients]
        server.writer.add_scalars(
            "global_models",
            {args["algorithm"]: len(set(cluster_identities))},
            _round
        )

        server.send_params(selected_clients)

        for client in selected_clients:
            client.train()

        server.aggregate_with_clustering(selected_clients)
        server.send_params(selected_clients)

        # evaluate selected clients
        # if _round % 1 == 0:
        #     local_accuracy = server.local_evaluate(selected_clients, _round)
        #     global_accuracy = server.global_evaluate(selected_clients, global_test_sets, _round)
        #     print(f"Round {_round} | Local accuracy: {local_accuracy} | Global accuracy: {global_accuracy}")

        for client in clients:
            # save the training dataset at the previous round
            client.prev_train_set = deepcopy(client.train_set)

    server.send_params(clients)  # all clients use the global model to test at the last round
    server.last_round_evaluate(clients, global_test_sets)
