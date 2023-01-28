import os
import math
import time
import string
import random
import itertools
import requests
import pprint
import subprocess
from collections import defaultdict

class Controller:

    def __init__(self, param_grid, is_large_model, nodes, train_all_shards, train_shard_path="root_path"):
        self.param_grid = param_grid
        self.is_large_model = is_large_model
        self.nodes = nodes
        self.train_shard_path = train_shard_path
        self.create_workers()
        self.n_workers = len(self.worker_on_node)
        self.train_all_shards = train_all_shards

    def find_combinations(self):
        param_keys = list(self.param_grid.keys())

        params_list = [self.param_grid[key] for key in param_keys]
        combinations = list(itertools.product(*params_list))

        self.param_combinations = []
        for comb in combinations:
            d = {}
            for i in range(len(comb)):
                d[param_keys[i]] = comb[i]
            self.param_combinations.append(d)

        return self.param_combinations
    
    def create_workers(self):
        # create a virtual worker for every GPU in the cluster
        self.worker_on_node = defaultdict(tuple)
        self.train_partitions = []
        self.node_with_workers = defaultdict(list)
        worker_id = 0
        for id, gpus in self.nodes.items():
            ngpus = len(gpus)
            for gpu_id in range(ngpus):
                self.train_partitions.append(os.path.join(self.train_shard_path, "gpu" + str(gpu_id)))
                self.worker_on_node[worker_id] = (id, gpu_id)
                self.node_with_workers[id].append(worker_id)
                worker_id += 1

    def check_finished(self, worker, exec_id):
        if os.path.exists("stats/check_finished/check_finished_" + str(worker) + ".txt"):    
            with open("stats/check_finished/check_finished_" + str(worker) + ".txt", "r") as f:
                s = f.read().split("\n")
            return exec_id in s
        return False

    def init_epoch(self):
        initial_msts = self.find_combinations()
        self.model_id_to_mst_mapping = {}
        current_msts = [(mst_id, mst) for mst_id, mst in enumerate(initial_msts)]
        for (mst_id, mst) in current_msts:
            self.model_id_to_mst_mapping[mst_id] = mst
        
        self.model_list = list(self.model_id_to_mst_mapping.keys())

        s = "Model ID: Model msts\n"
        for i in range(len(self.model_list)):
            s += str(self.model_list[i]) + " : " + pprint.pformat(initial_msts[i]) + "\n"

        print("Initial model configurations:", s)
        self.model_nworkers_trained = []
        self.model_on_worker = []
        for _, m in enumerate(self.model_list):
            self.model_on_worker.append(None)
            self.model_nworkers_trained.append(0)

        self.mw_pair = []
        for _ in range(len(self.model_list)):
            lis = []
            for j in range(self.n_workers):
                lis.append(False)
            self.mw_pair.append(lis)
        self.worker_running_model = [None] * self.n_workers
        self.exec_id_on_worker = [None] * self.n_workers
        
        self.model_on_node = [None] * len(self.model_list)

    def get_runnable_model(self, w):
        runnable_model = None
        model_list = self.model_list[::]
        random.shuffle(model_list)
        
        for m in model_list:
            # print("self.mw_pair[m][w]", m, w, self.mw_pair[m][w])
            if self.mw_pair[m][w] == False:
                if self.model_on_worker[m] == None:
                    runnable_model = m
                    # Largest Job First
                    if self.is_large_model[m]:
                        break
        # print("runnable model worker:", str(runnable_model), " ", str(w))
        return runnable_model
    
    def log_model_update(self, m, w):
        hydra_required = self.is_large_model[m]
        node = self.worker_on_node[w][0]
        if self.train_all_shards and hydra_required:
            trained_shards = [self.train_partitions[w] for w in self.node_with_workers[node]]
        else:
            trained_shards = [self.train_partitions[w]]
        s = "Trained model {} on worker {} on node {} on shards {}".format(m, w, node, trained_shards)
        with open("stats/model_updates/model_{}.txt".format(m), "a+") as f: 
            f.write(s + "\n" )
    
    def update_states(self, m, w, exec_id, completed):
        if completed:
            node_id = self.worker_on_node[w][0]
            self.model_on_worker[m] = None
            
            # updating for all workers part of a worker pair
            self.worker_running_model[w] = None
            self.exec_id_on_worker[w] = None
            
            # for all workers on the node update its mw_pair if the model is a large model
            # if model group is small model update only its mgw_pair
            
            model_size = "large" if self.is_large_model[m] else "small"
            print("Received", model_size, "model", str(m), "trained on node:", str(node_id), " trained on worker:", str(w))
            
            if self.train_all_shards and self.is_large_model[m]:
                for w in self.node_with_workers[node_id]:
                    self.mw_pair[m][w] = True
                self.model_nworkers_trained[m] += len(self.node_with_workers[node_id])
            else:
                self.mw_pair[m][w] = True
                self.model_nworkers_trained[m] += 1

            self.log_model_update(m, w)
        else:
            # print("Updating state", str(m), " ",str(w))
            node = self.worker_on_node[w][0]
            self.model_on_node[m] = node
            self.model_on_worker[m] = w
            self.worker_running_model[w] = m
            self.exec_id_on_worker[w] = exec_id

    def update_models_to_build(self, models_to_build):
        remove_models = set()
        for m in models_to_build:
            model_trained = True
            for w in self.worker_on_node:
                if not self.mw_pair[m][w]:
                    model_trained = False
                    break
            if (model_trained):
                remove_models.add(m)
        
        return models_to_build - remove_models

    def scheduler(self, epoch):
        models_to_build = set(self.model_list)
        while(len(models_to_build) > 0):
            for w in range(self.n_workers):
                # print(w, self.worker_running_model[w])
                if self.worker_running_model[w] == None:
                    m = self.get_runnable_model(w)
                    if m != None:
                        exec_id = self.launch_job(epoch, m, w)
                        self.update_states(m, w, exec_id, False)
                else:
                    exec_id = self.exec_id_on_worker[w]
                    if self.check_finished(w, exec_id):
                        print("Worker {} finished".format(w))
                        m = self.worker_running_model[w]
                        self.update_states(m, w, exec_id, True)
                        
                        models_to_build = self.update_models_to_build(models_to_build)

    def launch_job(self, epoch, m, w):
        hydra_required = self.is_large_model[m]
        node, gpu = self.worker_on_node[w]
        prev_node = self.model_on_node[m]
        
        if self.train_all_shards and hydra_required:
            train_shards = [self.train_partitions[w] for w in self.node_with_workers[node]]
        else:
            train_shards = [self.train_partitions[w]]
        
        s = "Scheduled epoch {} of model {} on worker {}".format(epoch, m, w)
        s1 = "with node {} on GPU {}".format(node, gpu)    
        s2 = "on train shards {}".format(str(train_shards))
        print(s + " " + s1 + " " + s2)
        
        exec_id = ''.join(random.choice(string.ascii_lowercase + string.digits) for _ in range(32))
        data = {
            "epoch": epoch,
            "model": m,
            "exec_id": str(exec_id),
            "w": w,
            "hydra_required": hydra_required,
            "train_shards": train_shards,
            "node": node,
            "prev_node": prev_node,
            "gpu": gpu
        }
        worker_ip = "http://localhost:" + str(8000 + node) + "/hydra"
        requests.post(url=worker_ip, json=data)

        return exec_id

def main():
    nlarge = 16
    nsmall = 0

    train_all_shards = True
    is_large_model = ([True]*nlarge) + ([False]*nsmall)
    
    nodes = {0: ["GPU0", "GPU1", "GPU2", "GPU3"],
           1: ["GPU0", "GPU1"],
           2: ["GPU0", "GPU1"]
        }
    
    param_grid = {
        'learning_rate': [1e-2, 1e-3],
        'embed_size': [256, 512],
        'hidden_size': [256, 512],
        'batch_size': [128, 256]
    }
    controller = Controller(param_grid, is_large_model, nodes, train_all_shards)
    controller.init_epoch()
    nepochs = 1
    for i in range(nepochs):
        controller.scheduler(i)

main()