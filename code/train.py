import argparse
import switch
from time import time, sleep

parser = argparse.ArgumentParser()

parser.add_argument('-model_save', action='store', dest='model_save', type=str,
                    help='model save folder ', required =True)

parser.add_argument('-instance_id', action='store', dest='instance_id', type=str,
                    help='instance_id ', required =True)

parser.add_argument('-model_recover', action='store', dest='model_recover', type=str,
                    help='model recover ', required =False, default=None)

parser.add_argument('-lr', action='store', dest='lr', type=float,
                    help='learning rate', required =False, default=0.0001)

parser.add_argument('-lr_decay', action='store', dest='lr_decay', type=float,
                    help='learning rate decay', required =False, default=0.5)

parser.add_argument('-lr_decay_step', action='store', dest='lr_decay_step', type=int,
                    help='learning rate decay step', required =False, default=50000)

parser.add_argument('-init_step', action='store', dest='init_step', type=int,
                    help='initial step size ', required =False, default=0)

parser.add_argument('-max_step', action='store', dest='max_step', type=int,
                    help='initial step size ', required =False, default=320001)

parser.add_argument('-mode', action='store', dest='mode', type=str,
                    help='mode [train][test][validate]', required =False, default="train")

parser.add_argument('-model', action='store', dest='model', type=str,
                    help='model [le][2dhist]', required =False, default="le")

parser.add_argument('-timesplit', action='store', dest='timesplit', type=int,
                    help='timesplit', required =False, default=2)

parser.add_argument('-cityname', action='store', dest='cityname', type=str,
                    help='cityname', required =False, default="la")

parser.add_argument('-exp', action='store', dest='expname', type=str,
                    help='expname', required =False, default="none")

parser.add_argument('-no_skip_connection', action='store', dest='no_skip_connection', type=str,
                    help='expname', required =False, default="none")

parser.add_argument('-no_dropout', action='store', dest='no_dropout', type=str,
                    help='expname', required =False, default="none")

parser.add_argument('-width', action='store', dest='width', type=int,
                    help='width', required =False, default=32)

args = parser.parse_args()

print(args)

if args.timesplit != 0:
    switch.USE_TEMPORAL_SPLIT = True

switch.USE_SHALLOW_DECODER = False

if args.expname == "dropout_gps_separately":
	switch.GPS_DROPOUT_SEPARATELY = True
	switch.TRAIN_ONLY_GPS_FEATURES = True
	print(args.expname)
	sleep(5.0)

if args.expname == "exp1":
	switch.GPS_DROPOUT_SEPARATELY = True
	print(args.expname)
	sleep(5.0)

crosscity = False
if args.expname == "crosscity":
	switch.TRAIN_ONLY_GPS_FEATURES = True
	switch.TRAIN_ONLY_ONE_MODEL = True
	crosscity = True
	print(args.expname)
	sleep(5.0)	

if args.expname == "capacity":
	switch.TRAIN_ONLY_GPS_FEATURES = True
	switch.TRAIN_ONLY_ONE_MODEL = True

	switch.MODEL_WIDTH = args.width 

	print(args.expname, switch.MODEL_WIDTH)
	sleep(5.0)	

dataset_ratio_n = 1
dataset_ratio_r = 1

if args.expname.startswith("datasize"):
	switch.TRAIN_ONLY_GPS_FEATURES = True
	switch.TRAIN_ONLY_ONE_MODEL = True

	switch.MODEL_WIDTH = 32

	items = args.expname.split("_")
	dataset_ratio_n = int(items[1])
	dataset_ratio_r = int(items[2])
	# if idx % n < r : keep else discard. 

	print(args.expname, dataset_ratio_n, dataset_ratio_r)
	sleep(5.0)	


if args.no_dropout == "true":
	switch.NO_DROPOUT = True 
	print("no dropout")
	sleep(5.0)

if args.no_skip_connection == "true":
	switch.NO_SKIP_CONNECTION = True 
	print("no skip connection")
	sleep(5.0)


from model import RiskModel, model_version
from dataloader import SingleTileDataLoader as DataLoader
from dataloader import checkAccidentsMap, ParallelDataLoader
from subprocess import Popen 
import numpy as np 

import tensorflow as tf 
from time import time 

import json 
import random 
import requests
from switch import USE_TEMPORAL_SPLIT 
from PIL import Image 


log_folder = "alllogs"

from datetime import datetime
instance_id = args.instance_id+"_timesplit"+str(args.timesplit) + "_cityname_"+args.cityname + "_width%d" % switch.MODEL_WIDTH + "_exp_" + args.expname
run = "run-"+datetime.today().strftime('%Y-%m-%d-%H-%M-%S')+"-"+instance_id

validation_folder = "validation_" + instance_id 
Popen("mkdir -p "+validation_folder, shell=True).wait()

model_save_folder = args.model_save + instance_id + "/"

Popen("mkdir -p %s" % model_save_folder, shell=True).wait()


datasetcfg = json.load(open("dataSample/dataset.cfg"))
datasetcfg_training = json.load(open("dataSample/dataset.cfg"))
datasetcfg_testing = json.load(open("dataSample/dataset.cfg"))

datasetcfg_training["alltiles"] = []
datasetcfg_testing["alltiles"] = []

for i in range(len(datasetcfg["alltiles"])):
	if crosscity:
		if i % 10 < 8:
			if datasetcfg["alltiles"][i][6] != args.cityname or args.cityname == "all":
				datasetcfg_training["alltiles"].append(datasetcfg["alltiles"][i])
		else:
			if datasetcfg["alltiles"][i][6] == args.cityname or args.cityname == "all":
				datasetcfg_testing["alltiles"].append(datasetcfg["alltiles"][i])
	else:
		if i % 10 < 8:
			if datasetcfg["alltiles"][i][6] == args.cityname or args.cityname == "all":
				datasetcfg_training["alltiles"].append(datasetcfg["alltiles"][i])
		else:
			if datasetcfg["alltiles"][i][6] == args.cityname or args.cityname == "all":
				datasetcfg_testing["alltiles"].append(datasetcfg["alltiles"][i])

training_set = []
for i in range(len(datasetcfg_training["alltiles"])):
	if i % dataset_ratio_n < dataset_ratio_r:
		training_set.append(datasetcfg_training["alltiles"][i])

datasetcfg_training["alltiles"] = training_set



print("training dataset %d tiles" % len(datasetcfg_training["alltiles"]))
print("testing dataset %d tiles" % len(datasetcfg_testing["alltiles"]))

epoch_size = int(len(datasetcfg_training["alltiles"]) / (48.0*48.0*4/400/400))
args.lr_decay_step = int(epoch_size * 20)
args.max_step = int(epoch_size * 50 + 2)

print("epoch size=%d lr_decay_step=%d max_step=%d save_interval=%d" % (epoch_size, args.lr_decay_step, args.max_step, epoch_size//2))
sleep(10.0)


step = args.init_step 
lr = args.lr
sum_loss = 0 

dataloader = ParallelDataLoader(datasetcfg=datasetcfg_training, batch_size=4, use_rotation=True, time_split = args.timesplit)
	
gpu_options = tf.GPUOptions(allow_growth=True)

with tf.Session(config=tf.ConfigProto(gpu_options = gpu_options)) as sess:
	model = RiskModel(sess, augch=1)
	writer = tf.summary.FileWriter(log_folder+"/"+run, sess.graph)
	if args.model_recover is not None:
		model.restoreModel(args.model_recover)
		print("restore model", args.model_recover)

	print("waiting for dataloader")
	dataloader.preload()
	print("start training")
	t0 = time() 

	t_train = 0 
	t_load = 0
	
	tmp_loss = 0

	perf_t = time() 

	nModel = len(model.names)
	tmp_loss = [0 for i in range(nModel)]


	while True:
		t0 = time()
		traces, batch_sat, batch_road, batch_gps2d, batch_target, batch_history, batch_features, batch_features_ds = dataloader.loadbatch()
		t1 = time() - t0
		t_load += t1 

		batchsize = 8

		trainingResults = model.Train(traces, batch_target, batch_sat, batch_road, batch_gps2d, batch_features, batch_features_ds, lr, history=batch_history)

		if step % batchsize == 0 and step > 0:
			model.ApplyGradient(lr)

		tmp_loss = [tmp_loss[i] + trainingResults[i] for i in range(nModel)]
		t_train += time() - t0

		if step % 10 == 0:
			print(step, "time all", t_train, "load batch", t_load, "loss", sum(tmp_loss))
			sum_loss += sum(tmp_loss) 

			t_train = 0 
			t_load = 0
			tmp_loss = [0 for i in range(nModel)]

		if step > -1 and step % 100 == 0:
			sum_loss /= 100 
			print(step, sum_loss, time() - t0)

			summary = model.addLog(sum_loss, sum_loss)
			writer.add_summary(summary, step)

			sum_loss = 0 

		if (step % (epoch_size*2) == 0): # and (step > 0) :
			model.saveModel(model_save_folder + "model_ep%d" % (step//epoch_size))

		if step % 100 == 0:
			Image.fromarray(((batch_sat[0,:,:,:]+0.5) *255.0).astype(np.uint8)).save(validation_folder+"/tile%d_sat.png" % ( (step/100)%32))
			Image.fromarray(((batch_road[0,:,:,0]+0.5) *255.0).astype(np.uint8)).save(validation_folder+"/tile%d_road.png" % ( (step/100)%32))
			Image.fromarray(((batch_gps2d[0,:,:,0]/10.0) *255.0).astype(np.uint8)).save(validation_folder+"/tile%d_gps2d.png" % ( (step/100)%32))
			
			def norm(x):
				maxv = np.amax(x)
				minv = np.amin(x)

				return (x-minv)/(maxv-minv+0.001)

			Image.fromarray((norm(batch_target[0,:,:,0])*255.0).astype(np.uint8)).save(validation_folder+"/tile%d_target.png" % ( (step/100)%32))
			Image.fromarray((norm(batch_history[0,:,:,0])*255.0).astype(np.uint8)).save(validation_folder+"/tile%d_history.png" % ( (step/100)%32))
			
			for i in range(nModel):
				#Image.fromarray(((np.clip(trainingResults[nModel+i][0,:,:,0]*10.0, 0.0, 1.0)) *255.0).astype(np.uint8)).save(validation_folder+"/tile%d_output%d.png" % ( (step/100)%32, i ) )
				Image.fromarray((norm(trainingResults[nModel+i][0,:,:,0]) *255.0).astype(np.uint8)).save(validation_folder+"/tile%d_output%d.png" % ( (step/100)%32, i ) )

				if model_version == "v10" and USE_TEMPORAL_SPLIT:
					# self.gate, self.hist, self.pred
					Image.fromarray((trainingResults[-3][0,:,:,0] *255.0).astype(np.uint8)).save(validation_folder+"/tile%d_gate%d.png" % ( (step/100)%32, i ) )
					Image.fromarray((norm(trainingResults[-3][0,:,:,0]) *255.0).astype(np.uint8)).save(validation_folder+"/tile%d_gatenorm%d.png" % ( (step/100)%32, i ) )
					Image.fromarray((norm(trainingResults[-2][0,:,:,0]) *255.0).astype(np.uint8)).save(validation_folder+"/tile%d_hist%d.png" % ( (step/100)%32, i ) )
					Image.fromarray((norm(trainingResults[-1][0,:,:,0]) *255.0).astype(np.uint8)).save(validation_folder+"/tile%d_pred%d.png" % ( (step/100)%32, i ) )

		if step % 20 == 0 and step != 0:
			dataloader.preload()

		if step > 0 and step % args.lr_decay_step == 0:
			lr = lr * args.lr_decay

		if step == args.max_step:
			break 


		if step % 200 == 0:
			print("training 200 iterations takes %.2f seconds" % (time()-perf_t))
			t200 = time()-perf_t

			print("ETC = ", (t200 * (args.max_step - step) // 200) / 3600.0, "hour(s)") 

			perf_t = time() 

		step += 1

















	