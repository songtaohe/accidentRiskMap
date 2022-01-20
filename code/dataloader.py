from time import time
import json  
import scipy.ndimage 
import scipy.misc
import numpy as np
import random 
import threading
from switch import USE_TEMPORAL_SPLIT
import pickle 
import switch

def checkAccidentsMap(datasetfolder = "", rid = 0):
	img = scipy.ndimage.imread(datasetfolder+"/region_%d_gt1.png" % rid)/255.0
	return np.sum(img)

def rotate(img, angle = 0, cval = 0):
	return scipy.ndimage.rotate(img, angle, reshape=False, cval=cval)

class SingleTileDataLoader():
	def __init__(self, datasetcfg = None, datasetfolder = "", tiles = [], batch_dim = 48, batch_size = 4, use_rotation = False, time_split = 2):
		self.datasetfolder = datasetfolder
		self.batch_dim = batch_dim
		self.batch_size = batch_size
		self.tile_dim = 400
		self.tiles = tiles 
		self.use_rotation = use_rotation 
		self.time_split = time_split
		self.datasetcfg = datasetcfg

		self.random_sat = np.random.rand(batch_dim*8, batch_dim*8, 3) - 0.5

	def preloadAsync(self, tid):
		self.t = threading.Thread(target=SingleTileDataLoader.preload, args=(self, tid)) 
		self.t.start()

	def preloadWait(self):
		self.t.join()

	def preload(self, tid):
		nAccidents = 0
		angle = 0 
		isTraining = True

		if tid is not None:
			if self.datasetcfg is not None:
				tile = self.datasetcfg["alltiles"][tid]
				self.datasetfolder = self.datasetcfg["metainfo"][tile[6]]["folder"]
			else:
				tile = self.tiles[tid]
				print("preloading", tid, tile[0])
				isTraining = False
		else:
			angle = random.randint(0,3) * 90 + random.randint(-30,30)

			if self.datasetcfg is not None:
				tile = random.choice(self.datasetcfg["alltiles"])
				self.datasetfolder = self.datasetcfg["metainfo"][tile[6]]["folder"]
			else:
				while nAccidents < 10:
					tile = random.choice(self.tiles)
					nAccidents = checkAccidentsMap(datasetfolder=self.datasetfolder, rid=tile[0])
					
		self.region = tile[1:]
		self.rid = tile[0]

		self.gps2d = np.log2(np.load(self.datasetfolder+"/region_%d_gps2d.npy" % self.rid) + 1.0)
		if isTraining and self.use_rotation:
			self.gps2d = rotate(self.gps2d, angle=angle)
			self.gps2d = np.clip(self.gps2d, 0.0, 100.0)
		
		self.gps_features = np.load(self.datasetfolder+"/region_%d_gps_features.npy" % self.rid)

		# rescale the gps features
		self.gps_features[:,:,0] = np.log2(self.gps_features[:,:,0] + 1.0)
		self.gps_features[:,:,1:4] = self.gps_features[:,:,1:4] / 50.0 
		self.gps_features[:,:,4:7] = self.gps_features[:,:,4:7] / 10.0
		self.gps_features[:,:,7:10] = self.gps_features[:,:,7:10] / 3.0 

		if isTraining and self.use_rotation:
			self.gps_features = rotate(self.gps_features, angle=angle)

		datasetfolder = self.datasetfolder 

		self.sat_image = scipy.ndimage.imread(datasetfolder+"/region_%d_sat.png" % self.rid)
		self.sat_image = self.sat_image.astype(np.float)/255.0 - 0.5 
		if isTraining and self.use_rotation:
			self.sat_image = rotate(self.sat_image, angle=angle)
	
		if isTraining:
			self.sat_image[:,:,0] = self.sat_image[:,:,0] * (0.7+0.6*random.random())
			self.sat_image[:,:,1] = self.sat_image[:,:,1] * (0.7+0.6*random.random())
			self.sat_image[:,:,2] = self.sat_image[:,:,2] * (0.7+0.6*random.random())
		else:
			pass
	
		self.sat_image = np.clip(self.sat_image, -0.5, 0.5)

		self.road_image = scipy.ndimage.imread(datasetfolder+"/region_%d_gt.png" % self.rid)
		self.road_image = self.road_image.astype(np.float)/255.0 - 0.5
		if isTraining and self.use_rotation:
			self.road_image = rotate(self.road_image, angle=angle, cval=-0.5)
			self.road_image = np.clip(self.road_image, -0.5, 0.5)

		self.history_input = scipy.ndimage.imread(datasetfolder+"/region_%d_t1_gt4.png" % self.rid).astype(np.float)/255.0
			
		if isTraining and self.use_rotation and USE_TEMPORAL_SPLIT:
			self.history_input = rotate(self.history_input, angle=angle)
			self.history_input = np.clip(self.history_input, 0.0, 1.0)

		self.target_image = scipy.ndimage.imread(datasetfolder+"/region_%d_t2_gt4.png" % self.rid).astype(np.float)/255.0
			
		if isTraining and self.use_rotation:
			self.target_image = rotate(self.target_image, angle=angle)
			self.target_image = np.clip(self.target_image, 0.0, 1.0)

		self.target_image_sum = np.sum(self.target_image)
		
		self.batch_sat = np.zeros((self.batch_size, self.batch_dim*8, self.batch_dim*8, 3))
		self.batch_road = np.zeros((self.batch_size, self.batch_dim*8, self.batch_dim*8, 1))
		self.batch_target = np.zeros((self.batch_size, self.batch_dim, self.batch_dim, 1))
		self.batch_gps2d = np.zeros((self.batch_size, self.batch_dim, self.batch_dim, 1))
		self.batch_gps_features = np.zeros((self.batch_size, self.batch_dim, self.batch_dim, 13))
		self.batch_history = np.zeros((self.batch_size, self.batch_dim, self.batch_dim, 1))
		
		print("preload done", tid)


	def loadbatch(self, x = None, y = None):
		
		for bid in range(self.batch_size):
			isTraining = False
			if x is None:
				isTraining = True
				cc = 0 
				while True:
					x = random.randint(24, self.tile_dim-24-self.batch_dim)
					y = random.randint(24, self.tile_dim-24-self.batch_dim)

					if np.sum(self.target_image[x:x+self.batch_dim, y:y+self.batch_dim]) > self.target_image_sum/20.0 :
						break

					cc += 1

					if cc > 10:
						break 

			
			self.batch_road[bid,:,:,0] = self.road_image[x*8:x*8+self.batch_dim*8, y*8:y*8+self.batch_dim*8]
			self.batch_target[bid,:,:,0] = self.target_image[x:x+self.batch_dim, y:y+self.batch_dim]
			
			if random.randint(0,100) >= 20 or isTraining == False or switch.NO_DROPOUT:
				self.batch_sat[bid,:,:,:] = self.sat_image[x*8:x*8+self.batch_dim*8, y*8:y*8+self.batch_dim*8,:]
			else:
				self.batch_sat[bid,:,:,:] = self.random_sat
			
			if switch.GPS_DROPOUT_SEPARATELY:
				if random.randint(0,100) >= 20 or isTraining == False or switch.NO_DROPOUT:
					self.batch_gps2d[bid,:,:,0] = self.gps2d[x:x+self.batch_dim, y:y+self.batch_dim]
				else:
					self.batch_gps2d[bid,:,:,0] = 0
				
				if random.randint(0,100) >= 20 or isTraining == False or switch.NO_DROPOUT:
					self.batch_gps_features[bid,:,:,:] = self.gps_features[x:x+self.batch_dim, y:y+self.batch_dim,:]	
				else:
					self.batch_gps_features[bid,:,:,:] = 0 
					
			else:
				if random.randint(0,100) >= 20 or isTraining == False or switch.NO_DROPOUT:
					self.batch_gps2d[bid,:,:,0] = self.gps2d[x:x+self.batch_dim, y:y+self.batch_dim]
					self.batch_gps_features[bid,:,:,:] = self.gps_features[x:x+self.batch_dim, y:y+self.batch_dim,:]
				else:
					self.batch_gps2d[bid,:,:,0] = 0 
					self.batch_gps_features[bid,:,:,:] = 0 
					

			if random.randint(0,100) >= 20 or isTraining == False or switch.NO_DROPOUT:
				self.batch_history[bid,:,:,0] = self.history_input[x:x+self.batch_dim, y:y+self.batch_dim]
			else:
				self.batch_history[bid,:,:,0] = 0 

			x = None # reset 

		return self.batch_sat, self.batch_road, self.batch_gps2d, self.batch_target, self.batch_history, self.batch_gps_features


class ParallelDataLoader():
	def __init__(self, *args,**kwargs):
		self.n = 4 
		self.subloader = []
		self.subloaderReadyEvent = []
		self.subloaderWaitEvent = []
		
		self.current_loader_id = 0 


		for i in range(self.n):
			self.subloader.append(SingleTileDataLoader(*args,**kwargs))
			self.subloaderReadyEvent.append(threading.Event())
			self.subloaderWaitEvent.append(threading.Event())

		for i in range(self.n):
			self.subloaderReadyEvent[i].clear()
			self.subloaderWaitEvent[i].clear()
		for i in range(self.n):
			x = threading.Thread(target=self.daemon, args=(i,))
			x.start() 


	def daemon(self, tid):
		c = 0

		while True:
			# 
			t0 = time()
			print("thread-%d starts preloading" % tid)
			self.subloader[tid].preload(None)
			
			self.subloaderReadyEvent[tid].set()

			print("thread-%d finished preloading (time = %.2f)" % (tid, time()-t0))

			self.subloaderWaitEvent[tid].wait()
			self.subloaderWaitEvent[tid].clear()

			if c == 0 and tid == 0:
				self.subloaderWaitEvent[tid].wait()
				self.subloaderWaitEvent[tid].clear()
			
			c = c + 1


	def preload(self):
		# release the current one 
		self.subloaderWaitEvent[self.current_loader_id].set()

		self.current_loader_id = (self.current_loader_id + 1) % self.n
		
		self.subloaderReadyEvent[self.current_loader_id].wait()
		self.subloaderReadyEvent[self.current_loader_id].clear()


	def loadbatch(self):
		return self.subloader[self.current_loader_id].loadbatch()		

	



