import os
import sys
import time
import random
import threading
import subprocess
from flask import Flask, request, jsonify
import datetime
import json


import logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

HYDRA_SLEEP = 30
MOP_SLEEP = 10

# network speed = 6.25 GB/s
# train time = 30x network time
NETWORK_SLEEP_LARGE = 0.3
NETWORK_SLEEP_SMALL = 0.1

app = Flask(__name__)
worker = None

def reset_stats():
    subprocess.run(["rm", "-rf", "stats"])
    os.mkdir("stats")
    os.mkdir("stats/check_finished")
    os.mkdir("stats/execution_time")
    os.mkdir("stats/model_updates")

def train_hydra(epoch, model, worker, exec_id, train_shards, node, prev_node, gpu):
    if prev_node != node:
        time.sleep(NETWORK_SLEEP_LARGE)
    
    st = time.time()
    # linear speed-up for hydra
    train_sleep = HYDRA_SLEEP * len(train_shards)
    time.sleep(train_sleep)
     

    now = datetime.datetime.now()
    en = time.time()
 
    print("Writing to check_finished", str(worker))
    with open("stats/check_finished/check_finished_" + str(worker) + ".txt", "a+") as f:
        f.write(str(exec_id) + "\n")
    execution_metrics_path = "stats/execution_time/worker_times" + str(worker) + ".json"
    execution_times = {}
    if(os.path.isfile(execution_metrics_path)):
        with open(execution_metrics_path, "r")  as f:
            execution_times = json.load(f)
    if ("epoch-" + str(epoch) not in execution_times):
        execution_times["epoch-" + str(epoch)] = {}
    execution_times["epoch-" + str(epoch)]["model-" + str(model)] = (st, en)
    with open(execution_metrics_path, "w+")  as f:
        json.dump(execution_times, f)
        
    print(str(now) + " Training of {} on node {} and GPU {} complete".format(str(model), node, gpu))
    print("Trained on datashards: {}".format(train_shards))

def train(epoch, model, worker, exec_id, train_shards, node, prev_node, gpu):
    if node != prev_node:
        time.sleep(NETWORK_SLEEP_SMALL)
    st = time.time()
    
    time.sleep(MOP_SLEEP)

    with open("stats/check_finished/check_finished_" + str(worker) + ".txt", "a+") as f:
        f.write(str(exec_id) + "\n")
    now = datetime.datetime.now()
    en = time.time()
    execution_metrics_path = "stats/execution_time/worker_times" + str(worker) + ".json"
    execution_times = {}
    if(os.path.isfile(execution_metrics_path)):
        with open(execution_metrics_path, "r")  as f:
            execution_times = json.load(f)
    if ("epoch-" + str(epoch) not in execution_times):
        execution_times["epoch-" + str(epoch)] = {}
    execution_times["epoch-" + str(epoch)]["model-" + str(model)] = (st, en)
    with open(execution_metrics_path, "w+")  as f:
        json.dump(execution_times, f)

    print(str(now) + " Training of {} on node {} and GPU(s) {} complete".format(str(model), node, gpu))
    print("Trained on datashards: {}".format(train_shards))

@app.route('/hello')
def hello_world():
    return "world"

@app.route('/hydra', methods=['POST'])
def hydra_on_worker():
    global worker

    data = request.get_json()
    
    epoch = data["epoch"]
    m = data["model"]
    w =  data["w"]
    exec_id = data["exec_id"]
    hydra_required = data["hydra_required"]
    train_shards = data["train_shards"]
    node = data["node"]
    prev_node = data["prev_node"]
    gpu = data["gpu"]

    if hydra_required:
        target_fn = train_hydra
    else:
        target_fn = train

    thread = threading.Thread(target=target_fn, args=(epoch, m, w, exec_id, train_shards, node, prev_node, gpu))
    thread.start()
    now = datetime.datetime.now()

    # print(str(now) + " Training models {} with ID {} on worker {} ; EPOCH {}".format(str(models), str(model_group_id), str(worker), str(epoch)))
    return "200 OK"

if __name__ == '__main__':
    worker = sys.argv[1]
    port = 8000 + int(worker)
    if int(worker) == 0:
        reset_stats()
    else:
        time.sleep(2)
    app.run(host="localhost", port=port, debug=True)