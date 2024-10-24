import yaml
import json
import time

from copy import deepcopy
from torch.nn.utils import vector_to_parameters
from entities.FedCCFA import FedCCFAClient, FedCCFAServer
from utils.gen_dataset import distribute_dataset
from utils.drift import sudden_drift, incremental_drift

if __name__ == "__main__":
    # read training parameters
    with open("../configs/FedCCFA.yaml", 'r') as f:
        args = yaml.load(f, Loader=yaml.FullLoader)
        print(json.dumps(args, indent=4))

    # initialize clients, server and global model
    client_train_set, client_test_set, global_test_sets = distribute_dataset(
        args["dataset"], args["client_num"], args["partition"], args["alpha"], args["seed"]
    )

    clients = []
    for client_id in range(args["client_num"]):
        client = FedCCFAClient(client_id, args, client_train_set[client_id], client_test_set[client_id], 0)
        clients.append(client)

    server = FedCCFAServer(args)
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

        selected_clients = server.select_clients(clients)
        server.send_params(selected_clients)

        balanced_clf_params_dict = {}

        for client in selected_clients:
            # considering that concept drift may occur at any round, get clients' label distributions at each round
            client.update_label_distribution()

            if args["balanced_epochs"] > 0:
                start_time = time.time()
                client.balance_train()
                balanced_clf_params_dict[client.id] = deepcopy(client.get_clf_parameters())
                end_time = time.time()
                total_time += end_time - start_time

            if not args["clustered_protos"]:
                client.global_protos = deepcopy(server.global_protos)

            client.train_with_protos(_round)
            if args["balanced_epochs"] == 0:
                balanced_clf_params_dict[client.id] = deepcopy(client.get_clf_parameters())

        server.aggregate_rep(selected_clients)
        server.aggregate_protos(selected_clients)
        server.send_rep_params(selected_clients)

        start_time = time.time()

        if server.args["oracle"]:
            label_merged_dict = server.oracle_merging(_round, [c.id for c in selected_clients])
        else:
            label_merged_dict = server.merge_classifiers(balanced_clf_params_dict)

        for label, merged_identities in label_merged_dict.items():
            print(label, merged_identities)
            for indices in merged_identities:
                # aggregate personalized classifier parameters according to label distribution
                clients_group = [client for client in selected_clients if client.id in indices]
                aggregated_label_params = server.aggregate_label_params(label, clients_group)
                aggregated_label_proto = server.aggregate_label_protos(label, clients_group)
                for client in clients_group:
                    client_label_params = [param[label] for name, param in client.model.named_parameters()
                                           if name in clf_keys]
                    vector_to_parameters(aggregated_label_params, client_label_params)
                    client.set_label_params(label, client_label_params)
                    client.global_protos[label] = aggregated_label_proto.clone()
        for client in selected_clients:
            client.p_clf_params = deepcopy(client.get_clf_parameters())

        end_time = time.time()
        total_time += end_time - start_time
        # print(total_time)

        # evaluate selected clients
        # if _round % 1 == 0:
        #     local_accuracy = server.local_evaluate(selected_clients, _round)
        #     global_accuracy = server.global_evaluate(selected_clients, global_test_sets, _round)
        #     print(f"Round {_round} | Local accuracy: {local_accuracy} | Global accuracy: {global_accuracy}")

    # fine-tune all clients' local models
    # for client in clients:
    #     client.fine_tune()

    server.last_round_evaluate(clients, global_test_sets)
