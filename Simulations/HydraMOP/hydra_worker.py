import os
import sys
import time
import random
import threading
from flask import Flask, request, jsonify
import datetime
import json


import logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

# 2 GPUs, 2 shards, 2 models
HYDRA_SLEEP = 5
# 1 GPU, 1 shard, 1 model
MOP_SLEEP = 2

app = Flask(__name__)
worker = None

# def create_empty_file():
# 	with open("check_finished_" + worker + ".txt", "w") as f:
# 		f.write("")

def train_hydra(epoch, models, wg, exec_id, model_group_id, train_shards, node, gpus):
	st = time.time()

	# linear speed-up for hydra
	train_sleep = HYDRA_SLEEP * (len(models)/2) * len(train_shards)
	time.sleep(train_sleep)

	now = datetime.datetime.now()
	en = time.time()
	for worker in wg:
		print("Writing to check_finished", str(worker), "wg:", str(wg))
		with open("stats/check_finished/check_finished_" + str(worker) + ".txt", "a+") as f:
			f.write(str(exec_id) + "\n")
		execution_metrics_path = "stats/execution_time/worker_times" + str(worker) + ".json"
		execution_times = {}
		if(os.path.isfile(execution_metrics_path)):
			with open(execution_metrics_path, "r")  as f:
				execution_times = json.load(f)
		if ("epoch-" + str(epoch) not in execution_times):
			execution_times["epoch-" + str(epoch)] = {}
		execution_times["epoch-" + str(epoch)]["model-group-" + str(model_group_id)] = (st, en)
		with open(execution_metrics_path, "w+")  as f:
			json.dump(execution_times, f)
		
	print(str(now) + " Training of {}  with ID {} on node {} and GPU(s) {} complete".format(str(models), str(model_group_id), node, gpus))
	print("Trained on datashards: {}".format(train_shards))

def train(epoch, models, wg, exec_id, model_group_id, train_shards, node, gpus):
	st = time.time()
	time.sleep(MOP_SLEEP)
	worker = wg[0]

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
	execution_times["epoch-" + str(epoch)]["model-group-" + str(model_group_id)] = (st, en)
	with open(execution_metrics_path, "w+")  as f:
		json.dump(execution_times, f)

	print(str(now) + " Training of {}  with ID {} on node {} and GPU(s) {} complete".format(str(models), str(model_group_id), node, gpus))
	print("Trained on datashards: {}".format(train_shards))

@app.route('/hello')
def hello_world():
	return "world"

@app.route('/hydra', methods=['POST'])
def hydra_on_worker():
	global worker

	data = request.get_json()
	
	epoch = data["epoch"]
	models = data["models"]
	wg =  data["wg"]
	exec_id = data["exec_id"]
	model_group_id = data["model_group_id"]
	hydra_required = data["hydra_required"]
	train_shards = data["train_shards"]
	node = data["node"]
	gpus = data["gpus"], 

	if hydra_required:
		target_fn = train_hydra
	else:
		target_fn = train

	thread = threading.Thread(target=target_fn, args=(epoch, models, wg, exec_id, model_group_id, train_shards, node, gpus))
	thread.start()
	now = datetime.datetime.now()

	# print(str(now) + " Training models {} with ID {} on worker {} ; EPOCH {}".format(str(models), str(model_group_id), str(worker), str(epoch)))
	return "200 OK"

if __name__ == '__main__':
	worker = sys.argv[1]
	port = 8000 + int(worker)
	app.run(host="localhost", port=port, debug=True)