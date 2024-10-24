import yaml
import json
import time

from copy import deepcopy
from utils.drift import sudden_drift, incremental_drift
from entities.FedPAC import FedPACClient, FedPACServer
from utils.gen_dataset import distribute_dataset
from torch.nn.utils import parameters_to_vector, vector_to_parameters

if __name__ == "__main__":
    # read training parameters
    with open("../configs/FedPAC.yaml", 'r') as f:
        args = yaml.load(f, Loader=yaml.FullLoader)
        print(json.dumps(args, indent=4))

    # initialize clients, server and global model
    client_train_set, client_test_set, global_test_sets = distribute_dataset(
        args["dataset"], args["client_num"], args["partition"], args["alpha"], args["seed"]
    )

    clients = []
    for client_id in range(args["client_num"]):
        client = FedPACClient(client_id, args, client_train_set[client_id], client_test_set[client_id], 0)
        clients.append(client)

    server = FedPACServer(args)
    server.get_client_data_size(clients)

    clf_keys = list(server.model.state_dict().keys())[-2:]

    for client in clients:
        client.clf_keys = clf_keys
    server.clf_keys = clf_keys

    for _round in range(args["rounds"]):
        total_time = 0
        if args["drift_pattern"] == "sudden" and _round == 100:
            sudden_drift(clients, global_test_sets, _round)
        elif args["drift_pattern"] == "recurrent" and _round in [100, 150]:
            sudden_drift(clients, global_test_sets, _round)
        elif args["drift_pattern"] == "incremental" and _round in [100, 110, 120]:
            incremental_drift(clients, global_test_sets, _round)

        Vars = []
        Hs = []

        selected_clients = server.select_clients(clients)
        server.send_rep_params(selected_clients)

        for client in selected_clients:
            # considering that concept drift may occur at any round, get clients' label distributions at each round
            client.update_label_distribution()

            # collect statistics
            start_time = time.time()
            v, h = client.statistics_extraction()
            Vars.append(deepcopy(v))
            Hs.append(deepcopy(h))
            end_time = time.time()
            total_time += end_time - start_time

            client.train_with_protos(server.global_protos)

        server.aggregate_rep(selected_clients)
        server.aggregate_protos(selected_clients)
        server.send_rep_params(selected_clients)

        # classifier combination
        start_time = time.time()
        avg_weights = server.get_head_agg_weight(Vars, Hs, len(selected_clients))
        all_clf_params = []
        for client in selected_clients:
            clf_params = [param for name, param in client.model.named_parameters() if name in clf_keys]
            all_clf_params.append(parameters_to_vector(clf_params))
        for index, client in enumerate(selected_clients):
            clf_params = [param for name, param in client.model.named_parameters() if name in clf_keys]
            new_clf_params = server.aggregate_clf(all_clf_params, avg_weights[index])
            vector_to_parameters(new_clf_params, clf_params)
        end_time = time.time()
        total_time += end_time - start_time
        # print(total_time)

        # evaluate selected clients
        # if _round % 1 == 0:
        #     local_accuracy = server.local_evaluate(selected_clients, _round)
        #     global_accuracy = server.global_evaluate(selected_clients, global_test_sets, _round)
        #     print(f"Round {_round} | Local accuracy: {local_accuracy} | Global accuracy: {global_accuracy}")

    server.last_round_evaluate(clients, global_test_sets)
