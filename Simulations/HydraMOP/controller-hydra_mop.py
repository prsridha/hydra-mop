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

    def __init__(self, param_grid, is_large_model, nodes, group_sz=2, train_shard_path="root_path"):
        self.param_grid = param_grid
        self.is_large_model = is_large_model
        self.group_sz = group_sz
        self.nodes = nodes
        self.train_shard_path = train_shard_path
        self.create_workers()
        self.n_workers = len(self.worker_on_node)

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
    
    def create_model_groups(self, models):
        large_models = []
        small_models = []
        for i in range(len(models)):
            if self.is_large_model[i]:
                large_models.append(models[i])
            else:
                small_models.append(models[i])
        
        nlm = len(large_models)
        nsm = len(small_models)
        ngroups = int(math.floor(nlm / self.group_sz))
        model_groups = []
        
        for i in range(ngroups):
            if i == ngroups - 1:
                model_groups.append(tuple(large_models[i*self.group_sz:]))
            else:
                model_groups.append(tuple(large_models[i*self.group_sz:(i + 1) * self.group_sz]))
            
        for i in range(nsm):
            model_groups.append(tuple((small_models[i],)))
        
        return model_groups
    
    def create_workers(self):
        # create a virtual worker for every GPU in the cluster
        self.worker_on_node = defaultdict(tuple)
        self.single_workers = []
        self.train_partitions = []
        self.node_with_workers = defaultdict(list)
        worker_id = 0
        for id, gpus in self.nodes.items():
            ngpus = len(gpus)
            if ngpus == 1:
                self.single_workers.append(worker_id)
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
        
        model_list = list(self.model_id_to_mst_mapping.keys())
        self.model_groups = self.create_model_groups(model_list)

        print("Model Groups:", self.model_groups)
        s = "Model ID: Model msts\n"
        for i in range(len(model_list)):
            s += str(model_list[i]) + " : " + pprint.pformat(initial_msts[i]) + "\n"

        print("Initial model configurations:", s)
        self.model_group_nworkers_trained = []
        self.model_group_on_workers = []
        for _, mg in enumerate(self.model_groups):
            self.model_group_on_workers.append(None)
            self.model_group_nworkers_trained.append(0)

        self.mgw_pair = []
        for _ in range(len(self.model_groups)):
            lis = []
            for j in range(self.n_workers):
                lis.append(False)
            self.mgw_pair.append(lis)
        self.worker_running_model_group = [None] * self.n_workers
        self.exec_id_on_worker = [None] * self.n_workers

    def get_idle_workers(self):
        idle_worker_groups = set()
        idle_worker_singles = set()
        for w, m in enumerate(self.worker_running_model_group):
            if m == None:
                idle_worker_singles.add((w, ))
        for tupi in idle_worker_singles:
            for tupj in idle_worker_singles:
                i = tupi[0]
                j = tupj[0]
                if i >= j:
                    continue
                # if 2 workers belong to the same node, only then group together.
                if self.worker_on_node[i][0] == self.worker_on_node[j][0]:
                    idle_worker_groups.add((i, j))
        # add single GPU nodes to idle_worker_groups
        for w in self.single_workers:
            if (w,) in idle_worker_singles:
                idle_worker_groups.add((w,))

        return idle_worker_groups, idle_worker_singles
    
    def get_runnable_mgw_pair(self, idle_worker_groups, idle_worker_singles):
        runnable_mgw_pair = None
        if len(idle_worker_groups) == 0 and len(idle_worker_singles) == 0:
            return runnable_mgw_pair

        model_id_list = list(range(len(self.model_groups)))
        
        random.shuffle(model_id_list)
        for mgid in model_id_list:
            mg = self.model_groups[mgid]
            # mg is running on some other worker(s)
            if self.model_group_on_workers[mgid] != None:
                continue
            if len(mg) > 1:
                # large model group
                for wg in idle_worker_groups:
                    # check if mg has already been trained on any worker part of wg.
                    trained_on_wg = False
                    for w in wg:
                        if self.mgw_pair[mgid][w] == True:
                            trained_on_wg = True
                            break
                    # we found our runnable model_group worker_group pair
                    if not trained_on_wg:
                        runnable_mgw_pair = (mgid, wg)
                        # print("runnable_mgw_pair", str(runnable_mgw_pair))
                        return runnable_mgw_pair
            else:
                if not runnable_mgw_pair:
                    # small model group
                    for wg in idle_worker_singles:
                        w = wg[0]
                        if self.mgw_pair[mgid][w] == False:
                            runnable_mgw_pair = (mgid, wg)
                            # print("runnable_mgw_pair", str(runnable_mgw_pair))
                            # return runnable_mgw_pair
        # print("runnable_mgw_pair", str(runnable_mgw_pair))
        return runnable_mgw_pair
    
    def log_model_group_update(self, mgid, wg):
        mg = self.model_groups[mgid]
        hydra_required = len(mg) > 1
        node = self.worker_on_node[wg[0]][0]
        if hydra_required:
            trained_shards = [self.train_partitions[w] for w in self.node_with_workers[node]]
        else:
            trained_shards = [self.train_partitions[wg[0]]]
        s = "Trained model group {} with mgid {} on worker group {} on node {} on shards {}".format(mg, mgid, wg, node, trained_shards)
        with open("stats/model_updates/model_{}.txt".format(mgid), "a+") as f: 
            f.write(s + "\n" )
    
    def update_states(self, mgid, wg, exec_id, completed):
        if completed:
            all_wg = self.model_group_on_workers[mgid]
            # getting the node on which this worker is
            node_id = self.worker_on_node[wg[0]][0]
            self.model_group_on_workers[mgid] = None
            # updating for all workers part of a worker pair
            for w in all_wg:
                self.worker_running_model_group[w] = None
                self.exec_id_on_worker[w] = None
            # for all workers on the node update its mgw_pair if the model group is large model
            # if model group is small model update only its mgw_pair
            if (len(self.model_groups[mgid]) > 1):
                print("Received large model group", str(self.model_groups[mgid]), "with mgid:", str(mgid), "trained on node:", str(node_id), " worker ids trained on:", str(all_wg))
                for w in self.node_with_workers[node_id]:
                    self.mgw_pair[mgid][w] = True
                self.model_group_nworkers_trained[mgid] += len(self.node_with_workers[node_id])
            else:
                print("Received small model group", str(self.model_groups[mgid]), "with mgid:", str(mgid), "trained on node:", str(node_id), " worker ids trained on:", str(all_wg))
                self.mgw_pair[mgid][wg[0]] = True
                self.model_group_nworkers_trained[mgid] += 1
            self.log_model_group_update(mgid, all_wg)
        else:
            self.model_group_on_workers[mgid] = wg
            for w in wg:
                self.worker_running_model_group[w] = mgid
                self.exec_id_on_worker[w] = exec_id
            
    def poll_and_update_workers(self):
        for w in self.worker_on_node:
            exec_id = self.exec_id_on_worker[w]
            if exec_id:
                
                if self.check_finished(w, exec_id):
                    print("Worker {} finished".format(w))
                    mgid = self.worker_running_model_group[w]
                    wg = (w,)
                    self.update_states(mgid, wg, None, True)
    
    def update_model_groups_to_build(self, model_groups_to_build):
        remove_models = set()
        for mgid in model_groups_to_build:
            model_group_trained = True
            for w in self.worker_on_node:
                if not self.mgw_pair[mgid][w]:
                    model_group_trained = False
                    break
            if (model_group_trained):
                remove_models.add(mgid)
        
        return model_groups_to_build - remove_models
    
    def scheduler(self, epoch):
        model_groups_to_build = set(range(len(self.model_groups)))
        while(len(model_groups_to_build) > 0):
            idle_worker_groups, idle_worker_singles = self.get_idle_workers()
            # print(idle_worker_groups, idle_worker_singles)
            runnable_mgw_pair = self.get_runnable_mgw_pair(idle_worker_groups, idle_worker_singles)
            # print(runnable_mgw_pair)
            if runnable_mgw_pair:
                mgid, wg = runnable_mgw_pair
                # print(runnable_mgw_pair)
                exec_id = self.launch_job(epoch, mgid, self.model_groups[mgid], wg)
                self.update_states(mgid, wg, exec_id, False)
            self.poll_and_update_workers()
            model_groups_to_build = self.update_model_groups_to_build(model_groups_to_build)
            # time.sleep(1)

    def launch_job(self, epoch, mgid, mg,  wg):
        hydra_required = len(mg) > 1
        s = "Scheduled epoch {} of model_group {} with mgid {} on worker group {}".format(epoch, mg, mgid, wg)
        gpus = [self.worker_on_node[w][1] for w in wg]
        node = self.worker_on_node[wg[0]][0]
        s1 = "with node {} on GPU {}".format(node, gpus)
        if hydra_required:
            train_shards = [self.train_partitions[w] for w in self.node_with_workers[node]]
        else:
            train_shards = [self.train_partitions[wg[0]]]
            
        s2 = "on train shards {}".format(str(train_shards))
        
        print(s + " " + s1 + " " + s2)
        
        exec_id = ''.join(random.choice(string.ascii_lowercase + string.digits) for _ in range(32))
        data = {
            "epoch": epoch,
            "models": list(mg),
            "exec_id": str(exec_id),
            "wg": list(wg),
            "model_group_id": str(mgid),
            "hydra_required": hydra_required,
            "train_shards": train_shards,
            "node": node,
            "gpus": gpus
        }
        worker_ip = "http://localhost:" + str(8000 + node) + "/hydra"
        requests.post(url=worker_ip, json=data)

        return exec_id
    

def main():
    nlarge = 5
    nsmall = 11

    is_large_model = ([True]*nlarge) + ([False]*nsmall)
    
    nodes = {0: ["GPU0", "GPU1","GPU2","GPU3"],
           1: ["GPU0", "GPU1"],
           2: ["GPU0", "GPU1"]
        }
    
    param_grid = {
        'learning_rate': [1e-2, 1e-3],
        'embed_size': [256, 512],
        'hidden_size': [256, 512],
        'batch_size': [128, 256]
    }
    controller = Controller(param_grid, is_large_model, nodes)
    controller.init_epoch()
    nepochs = 1
    for i in range(nepochs):
        controller.scheduler(i)

main()