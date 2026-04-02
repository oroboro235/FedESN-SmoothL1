
FL_exp_config = {
    "n_rounds": 5,
    "n_clients": 5,
    "seed": 1234,
    # "reg_type": "none",
    "global_lr": 1.0,
    "local_lr": 1.0,
    # local run
    "local_epochs": 100,
}
import config
import os
import time
import json
from logger import myLogger

import numpy as np


np.random.seed(FL_exp_config["seed"])

from client import Client_TSC
from server import Server_FedAvg


# hyperparameters settings:
settings = []
def read_settings(reg_type: str):
    name_jsonFile = "FL_settings_"+reg_type+".json"
    with open(os.path.join(name_jsonFile), "r") as f:
        settings = json.load(f)
    return settings




def shuffle_data(Xtr, ytr, Xte, yte):
    # shuffle data
    idx = np.random.permutation(len(Xtr))
    Xtr, ytr = Xtr[idx], ytr[idx]
    idx = np.random.permutation(len(Xte))
    Xte, yte = Xte[idx], yte[idx]
    return Xtr, ytr, Xte, yte
    
def data_preprocessing(Xtr, ytr, Xte, yte, setting={}, isShuffle=False):

    # check if y is one-hot encoded
    if len(ytr.shape) == 1:
        ytr = np.eye(setting["output_dim"])[ytr.astype(int)]
        yte = np.eye(setting["output_dim"])[yte.astype(int)]
    
    # shuffle data
    if isShuffle:
        Xtr, ytr, Xte, yte = shuffle_data(Xtr, ytr, Xte, yte)

    return Xtr, ytr, Xte, yte

def data_partition(Xtr, ytr, Xte, yte, n_clients, isOverlapping=True):

    Xtr_split = []
    ytr_split = []
    Xte_split = []
    yte_split = []

    nSample_tr_perClient = len(Xtr)//n_clients
    nSample_te_perClient = len(Xte)//n_clients

    if isOverlapping:
        for i in range(n_clients):
            # idx = np.arange(i*nSample_tr_perClient, (i+1)*nSample_tr_perClient)
            idx = np.random.permutation(len(Xtr))[:nSample_tr_perClient]
            Xtr_split.append(Xtr[idx])
            ytr_split.append(ytr[idx])
            # idx = np.arange(i*nSample_te_perClient, (i+1)*nSample_te_perClient)
            idx = np.random.permutation(len(Xte))[:nSample_te_perClient]
            Xte_split.append(Xte[idx])
            yte_split.append(yte[idx])
    else:
        for i in range(n_clients):
            idx = np.arange(i, len(Xtr), n_clients)
            Xtr_split.append(Xtr[idx])
            ytr_split.append(ytr[idx])
            idx = np.arange(i, len(Xte), n_clients)
            Xte_split.append(Xte[idx])
            yte_split.append(yte[idx])
    
    return Xtr_split, ytr_split, Xte_split, yte_split

clients = []
def train_func(client_id):
    result = clients[client_id].train()
    clients[client_id].model.reset()
    return result, client_id
def evaluate_func(client_id):
    result = clients[client_id].evaluate()
    clients[client_id].model.reset()
    return result

def print_results(results, lg):
    print("ID\tAcc\tSparsity")
    lg.info("ID\tAcc\tSparsity")
    for result in results:
        print("{}\t{:.2f}%\t{:.2f}%".format(result["client_id"], result["acc"], result["sparsity"]))
        lg.info("{}\t{:.2f}%\t{:.2f}%".format(result["client_id"], result["acc"], result["sparsity"]))

def print_log(_str, lg):
    print(_str)
    lg.info(_str)



def main(reg_type: str):
    FL_exp_config["reg_type"] = reg_type
    settings = read_settings(FL_exp_config["reg_type"])

    # settings = settings[3:]
    settings = [settings[2]]

    for setting in settings:

        setting["seed"] = FL_exp_config["seed"]
        setting["reg_type"] = FL_exp_config["reg_type"]
        setting["global_lr"] = FL_exp_config["global_lr"]
        setting["local_lr"] = FL_exp_config["local_lr"]
        setting["local_epochs"] = FL_exp_config["local_epochs"]

        exp_name = f"{setting['dataset']}_{setting['reg_type']}"
        file_name = f"{exp_name}_{time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())}.log"
        _lg = myLogger(exp_name, file_name)


        print_log(f"Training on {setting['dataset']} dataset, regulization type: {setting['reg_type']}", _lg)

        # initializing server
        server = Server_FedAvg(hyperparams=setting)
        server.initialize_global_model()

        print_log("Initialized global model.", _lg)


        # load data and partition

        # data cache
        # check if the data is already loaded in cache folder
        if not os.path.exists(config.path.cache_path+setting["dataset"]+".npz"):
            from data_loader import read_data
            Xtr, ytr, Xte, yte = read_data(setting["dataset"])
            np.savez(config.path.cache_path+setting["dataset"], Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte)
        else:
            data = np.load(config.path.cache_path+setting["dataset"]+".npz")
            Xtr, ytr, Xte, yte = data["Xtr"], data["ytr"], data["Xte"], data["yte"]

        
        # shuffle
        Xtr, ytr, Xte, yte = data_preprocessing(Xtr, ytr, Xte, yte, 
                                                setting=setting, 
                                                isShuffle=False)
        # partition
        Xtr_split, ytr_split, Xte_split, yte_split = data_partition(Xtr, ytr, Xte, yte, 
                                                                    n_clients=FL_exp_config["n_clients"], 
                                                                    isOverlapping=False)

        # initializing clients
        clients.clear()
        for i in range(FL_exp_config["n_clients"]):
            client = Client_TSC(i, 
                                data=[Xtr_split[i], ytr_split[i], Xte_split[i], yte_split[i]], 
                                seed=setting["seed"])
            client.receive_hyperparams(setting)
            client.initialize_model(setting["output_dim"], server.res.Win, server.res.W)
        
            clients.append(client)
        
        print_log("Initialized {} clients. Each client has {} training samples and {} testing samples.".format(len(clients), len(Xtr_split[0]), len(Xte_split[0])), _lg)

        # run federated learning rounds
        print_log("Starting federated learning rounds...", _lg)


        # train in local
        from parallelbar import progress_map
        for i in range(FL_exp_config["n_rounds"]):
            print_log(f"Round {i+1}", _lg)
            print_log("Training clients...", _lg)


            # import tqdm
            # collected_clientWout = []
            # for client in tqdm.tqdm(clients):
            #     result = client.train()
            #     collected_clientWout.append(result)
            #     client.model.reset()


            results = progress_map(train_func, range(FL_exp_config["n_clients"]))
            collected_clientWout = []
            for (_Wout, cid) in results:
                clients[cid].model.nodes[1].Wout = _Wout.toarray()
                collected_clientWout.append(_Wout)


            # evaluate
            print_log("Before aggregation:", _lg)

            evaluate_results = progress_map(evaluate_func, range(FL_exp_config["n_clients"]))
            print_results(evaluate_results, _lg)


            # aggregate parameters
            global_Wout = server.aggregate_parameters(collected_clientWout)

            # client update local params with global params
            for client in clients:
                client.receive_parameters(global_Wout)

            print_log("After aggregation:", _lg)

            evaluate_results = progress_map(evaluate_func, range(FL_exp_config["n_clients"]))
            print_results(evaluate_results, _lg)
        
        # save global model
        np.savez(os.path.join(config.path.model_path, f"global_{exp_name}.npz"), Win=server.res.Win.toarray(), Wres=server.res.W.toarray(), Wout=global_Wout.toarray())
        _lg.info(f"Global model saved to {config.path.model_path}.")

        # break

if __name__ == '__main__':
    # for reg_type in ["none", "l2", "sl1"]:
    for reg_type in ["none"]:
        main(reg_type)