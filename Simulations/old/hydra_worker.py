import sys
import time
import random
import threading
from flask import Flask, request, jsonify


import logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)


app = Flask(__name__)
worker = None

def create_empty_file():
	with open("check_finished_" + worker + ".txt", "w") as f:
		f.write("")

def train(exec_id, models):
	time.sleep(random.randint(1,5))
	with open("check_finished_" + worker + ".txt", "a+") as f:
		f.write(str(exec_id) + "\n")
	print("Training of {} complete".format(models))

@app.route('/hello')
def hello_world():
	return "world"

@app.route('/hydra', methods=['POST'])
def hydra_on_worker():
	global worker

	data = request.get_json()
	
	epoch = data["epoch"]
	models = data["models"]
	exec_id = data["exec_id"]

	thread = threading.Thread(target=train, args=(exec_id, models))
	thread.start()

	print("Training models {} on worker {} ; EPOCH {}".format(str(models), str(worker), str(epoch)))
	return "200 OK"


if __name__ == '__main__':
	worker = sys.argv[1]
	port = 8000 + int(worker)
	create_empty_file()
	app.run(host="localhost", port=port, debug=True)