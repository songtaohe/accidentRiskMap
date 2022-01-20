import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import batch_norm
from time import time 
import tf_common_layer as common

from switch import USE_TEMPORAL_SPLIT, USE_SHALLOW_DECODER, MODEL_WIDTH
import switch

from resnet import resblock as residual_block
from resnet import relu
from resnet import batch_norm as batch_norm_resnet  

model_version = "v10"

class RiskModel():
	def __init__(self, sess, augch = 4, input_dim = 48):
		t0 = time()

		self.sess = sess 
		self.input_dim = input_dim 
		self.batchsize = 1 

		self.augch = augch # 1 or 4 
		
		def aug4(x):
			return tf.concat([x, tf.image.rot90(x,k=1), tf.image.rot90(x,k=2), tf.image.rot90(x,k=3)], axis=0)

		self.target_in = tf.placeholder(tf.float32, shape = [None,self.input_dim, self.input_dim, 1])
		self.history_in = tf.placeholder(tf.float32, shape = [None,self.input_dim, self.input_dim, 1])

		if self.augch == 4:
			self.target = aug4(self.target_in)
		else:
			self.target = self.target_in

		if self.augch == 4:
			self.history = aug4(self.history_in)
		else:
			self.history = self.history_in


		self.lr = tf.placeholder(tf.float32, shape=[])
		self.is_training = tf.placeholder(tf.bool)

		####################  INPUT IMAGERY ###################
		self.input_sat_in = tf.placeholder(tf.float32, shape = [None,self.input_dim*8, self.input_dim*8, 3])
		self.input_road_seg_in = tf.placeholder(tf.float32, shape = [None,self.input_dim*8, self.input_dim*8, 1])

		if self.augch == 4:
			self.input_sat = aug4(self.input_sat_in)
			self.input_road_seg = aug4(self.input_road_seg_in)
		else:
			self.input_sat = self.input_sat_in
			self.input_road_seg = self.input_road_seg_in


		####################  INPUT GPS 2D HISTOGRAM ###################
		self.input_gps_2d_in = tf.placeholder(tf.float32, shape = [None,self.input_dim, self.input_dim, 1])
		
		if self.augch == 4:
			self.input_gps_2d = aug4(self.input_gps_2d_in)
		else:
			self.input_gps_2d = self.input_gps_2d_in 

		
		####################  INPUT GPS Feature HISTOGRAM ###################
		self.input_gps_features_in = tf.placeholder(tf.float32, shape = [None,self.input_dim, self.input_dim, 13])
		
		if self.augch == 4:
			self.input_gps_features = aug4(self.input_gps_features_in)
		else:
			self.input_gps_features = self.input_gps_features_in


		### Create different models ###
		self.losses = []
		self.outputs = []
		self.trainops = []
		self.names = []

		v1 = False

		# 0
		if switch.TRAIN_ONLY_GPS_FEATURES == False:
			tag = "2DGPS_SAT_ROAD"
			with tf.variable_scope(tag):
				hires_in = tf.concat([self.input_sat, self.input_road_seg], axis=3) 
				hires_ebd = self.buildEncoderWide(hires_in, ch_in=4, ch = 32, ch_out=63)
				lowres_ebd = self.input_gps_2d

				if USE_TEMPORAL_SPLIT:
					print("use temporal split")
					ebd = tf.concat([hires_ebd, lowres_ebd], axis=3)
					out = self.buildFuseNet(ebd, self.history, ch_in=32+32, ch=32)
				else:
					ebd = tf.concat([hires_ebd, lowres_ebd], axis=3)
					out = self.buildUNET(ebd, ch_in=32+32, ch=32)

				loss = tf.reduce_mean(tf.square(out-self.target))
				trainop = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(loss)

				self.names.append(tag)
				self.outputs.append(out)
				self.losses.append(loss)
				self.trainops.append(trainop)

		# 1
		if switch.TRAIN_ONLY_GPS_FEATURES == False:
			tag = "SAT_ROAD"
			with tf.variable_scope(tag):
				hires_in = tf.concat([self.input_sat, self.input_road_seg], axis=3) 
				hires_ebd = self.buildEncoderWide(hires_in, ch_in=4, ch = 32, ch_out=64)

				if USE_TEMPORAL_SPLIT:
					print("use temporal split")
					ebd = tf.concat([hires_ebd], axis=3)
					out = self.buildFuseNet(ebd, self.history, ch_in=64, ch=32)
				else:
					ebd = hires_ebd
					out = self.buildUNET(ebd, ch_in=64, ch=32)
				
				loss = tf.reduce_mean(tf.square(out-self.target))
				trainop = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(loss)

				self.names.append(tag)
				self.outputs.append(out)
				self.losses.append(loss)
				self.trainops.append(trainop)

		# 2
		if switch.TRAIN_ONLY_GPS_FEATURES == False:
			tag = "2DGPS_ROAD"
			with tf.variable_scope(tag):
				hires_in = self.input_road_seg
				hires_ebd = self.buildEncoderWide(hires_in, ch_in=1, ch=32, ch_out=63)

				lowres_ebd = self.input_gps_2d

				if USE_TEMPORAL_SPLIT:
					print("use temporal split")
					ebd = tf.concat([hires_ebd, lowres_ebd], axis=3)
					out = self.buildFuseNet(ebd, self.history, ch_in=64, ch=32)
				else:
					ebd = tf.concat([hires_ebd, lowres_ebd], axis=3)
					out = self.buildUNET(ebd, ch_in=64, ch=32)

				loss = tf.reduce_mean(tf.square(out-self.target))
				trainop = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(loss)

				self.names.append(tag)
				self.outputs.append(out)
				self.losses.append(loss)
				self.trainops.append(trainop)

		# 3
		if switch.TRAIN_ONLY_GPS_FEATURES == False:
			tag = "ROAD"
			with tf.variable_scope(tag):
				hires_in = self.input_road_seg
				hires_ebd = self.buildEncoderWide(hires_in, ch_in=1, ch = 32, ch_out=64)

				if USE_TEMPORAL_SPLIT:
					print("use temporal split")
					ebd = tf.concat([hires_ebd], axis=3)
					out = self.buildFuseNet(ebd, self.history, ch_in=64, ch=32)
				else:
					ebd = hires_ebd
					out = self.buildUNET(ebd, ch_in=64, ch=32)
				loss = tf.reduce_mean(tf.square(out-self.target))
				trainop = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(loss)

				self.names.append(tag)
				self.outputs.append(out)
				self.losses.append(loss)
				self.trainops.append(trainop)

		# 4		
		tag = "GPS_FEATURES_SAT_ROAD"
		with tf.variable_scope(tag):
			nChannel = MODEL_WIDTH

			hires_in = tf.concat([self.input_sat, self.input_road_seg], axis=3) 
			hires_ebd = self.buildEncoderWide(hires_in, ch_in=4, ch_out=nChannel)
			
			if switch.GPS_DROPOUT_SEPARATELY:
				lowres_ebd = self.buildEncoderV2(self.input_gps_features, ch_out=nChannel-1)
				ebd = tf.concat([hires_ebd, lowres_ebd, self.input_gps_2d], axis=3)
			else:
				lowres_ebd = self.buildEncoderV2(self.input_gps_features, ch_out=nChannel)
				ebd = tf.concat([hires_ebd, lowres_ebd], axis=3)
			
			if USE_TEMPORAL_SPLIT:
				print("use temporal split")
				out = self.buildFuseNet(ebd, self.history, ch_in=nChannel+nChannel, ch=nChannel)
			else:
				out = self.buildUNET(ebd, ch_in=nChannel+nChannel, ch=nChannel)

			loss = tf.reduce_mean(tf.square(out-self.target))
			trainop = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(loss)

			self.names.append(tag)
			self.outputs.append(out)
			self.losses.append(loss)
			self.trainops.append(trainop)
		
		# 5
		if switch.TRAIN_ONLY_ONE_MODEL == False:
			tag = "GPS_FEATURES_ROAD"
			with tf.variable_scope(tag):
				hires_in = self.input_road_seg
				hires_ebd = self.buildEncoderWide(hires_in, ch_in=1, ch_out=32)

				if switch.GPS_DROPOUT_SEPARATELY:
					lowres_ebd = self.buildEncoderV2(self.input_gps_features, ch_out=31)
					ebd = tf.concat([hires_ebd, lowres_ebd, self.input_gps_2d], axis=3)
				else:
					lowres_ebd = self.buildEncoderV2(self.input_gps_features)
					ebd = tf.concat([hires_ebd, lowres_ebd], axis=3)

				if USE_TEMPORAL_SPLIT:
					print("use temporal split")
					out = self.buildFuseNet(ebd, self.history, ch_in=32+32, ch=32)
				else:
					out = self.buildUNET(ebd, ch_in=32+32, ch=32)

				loss = tf.reduce_mean(tf.square(out-self.target))
				trainop = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(loss)

				self.names.append(tag)
				self.outputs.append(out)
				self.losses.append(loss)
				self.trainops.append(trainop)

		

		

		print("create models and optimizers", time() - t0)

		self.sess.run(tf.global_variables_initializer())

		print("initialize weights", time() - t0)

		self.saver = tf.train.Saver(max_to_keep=30)

		self.summary_loss = []
		
		self.test_loss =  tf.placeholder(tf.float32)
		self.train_loss =  tf.placeholder(tf.float32)

		self.summary_loss.append(tf.summary.scalar('loss/test', self.test_loss))
		self.summary_loss.append(tf.summary.scalar('loss/train', self.train_loss))

		self.merged_summary = tf.summary.merge_all()

		print("done", time() - t0)

	def buildEncoder(self, x, ch_in = 3, ch_out=64):
		x, _, _ = common.create_conv_layer('cnn_enc_l1', x, ch_in, ch_out // 4, kx = 3, ky = 3, stride_x = 1, stride_y = 1, is_training = self.is_training, batchnorm = True)
		x, _, _ = common.create_conv_layer('cnn_enc_l2', x, ch_out // 4, ch_out // 2, kx = 3, ky = 3, stride_x = 2, stride_y = 2, is_training = self.is_training, batchnorm = True)

		x, _, _ = common.create_conv_layer('cnn_enc_l3', x, ch_out // 2, ch_out // 2, kx = 3, ky = 3, stride_x = 1, stride_y = 1, is_training = self.is_training, batchnorm = True)
		x, _, _ = common.create_conv_layer('cnn_enc_l4', x, ch_out // 2, ch_out, kx = 3, ky = 3, stride_x = 2, stride_y = 2, is_training = self.is_training, batchnorm = True)

		x, _, _ = common.create_conv_layer('cnn_enc_l5', x, ch_out , ch_out , kx = 3, ky = 3, stride_x = 1, stride_y = 1, is_training = self.is_training, batchnorm = True)
		x, _, _ = common.create_conv_layer('cnn_enc_l6', x, ch_out , ch_out, kx = 3, ky = 3, stride_x = 2, stride_y = 2, is_training = self.is_training, batchnorm = True)

		return x 

	def buildEncoderWide(self, x, ch_in = 3, ch = 32, ch_out=32):
		x, _, _ = common.create_conv_layer('cnn_enc_l1', x, ch_in, ch, kx = 3, ky = 3, stride_x = 1, stride_y = 1, is_training = self.is_training, batchnorm = True)
		x, _, _ = common.create_conv_layer('cnn_enc_l2', x, ch, ch, kx = 3, ky = 3, stride_x = 2, stride_y = 2, is_training = self.is_training, batchnorm = True)

		x, _, _ = common.create_conv_layer('cnn_enc_l3', x, ch, ch, kx = 3, ky = 3, stride_x = 1, stride_y = 1, is_training = self.is_training, batchnorm = True)
		x, _, _ = common.create_conv_layer('cnn_enc_l4', x, ch, ch, kx = 3, ky = 3, stride_x = 2, stride_y = 2, is_training = self.is_training, batchnorm = True)

		x, _, _ = common.create_conv_layer('cnn_enc_l5', x, ch , ch , kx = 3, ky = 3, stride_x = 1, stride_y = 1, is_training = self.is_training, batchnorm = True)
		x, _, _ = common.create_conv_layer('cnn_enc_l6', x, ch , ch_out, kx = 3, ky = 3, stride_x = 2, stride_y = 2, is_training = self.is_training, batchnorm = True)

		return x 

	def buildEncoderV2(self, x, ch_in = 13, ch_out=32):
		x, _, _ = common.create_conv_layer('cnn_encv2_l1', x, ch_in, ch_out, kx = 1, ky = 1, stride_x = 1, stride_y = 1, is_training = self.is_training, batchnorm = True)
		x, _, _ = common.create_conv_layer('cnn_encv2_l2', x, ch_out, ch_out, kx = 1, ky = 1, stride_x = 1, stride_y = 1, is_training = self.is_training, batchnorm = True)

		return x

	def buildFuseNet(self, x, hist, ch_in = 13, ch_out=32, ch=32):
		x_ = tf.concat([x, hist], axis=3)
		if USE_SHALLOW_DECODER:
			x = self.buildShallowDecoder(x_, ch_in = ch_in + 1, ch_out = 2)
		else:
			x = self.buildUNET_resnet18(x_, ch_in = ch_in + 1, ch_out=2, ch=ch_in + 1)

		print("resnet18 channel ", ch_in, ch_out, ch)
		
		gate = tf.nn.sigmoid(x[:,:,:,0:1])
		pred = x[:,:,:,1:2]

		hist, _, _ = common.create_conv_layer('cnn_hist_l1', hist, 1, 1, kx = 7, ky = 7, stride_x = 1, stride_y = 1, is_training = self.is_training, batchnorm = False, activation = "linear")
		
		self.gate = gate 
		self.hist = hist 
		self.pred = pred 

		if switch.NO_SKIP_CONNECTION:
			return pred
		else:
			return tf.multiply(pred, gate) + tf.multiply(hist, 1-gate) 

	def buildUNET(self, x, ch_in = 16, ch = 32):
		if USE_SHALLOW_DECODER:
			ret = self.buildShallowDecoder(x, ch_in = ch_in, ch_out = 1)
		else:
			ret = self.buildUNET_resnet18(x, ch_in = ch_in, ch = ch_in)

		return ret 


	def buildShallowDecoder(self, x, ch_in = 16, ch = 32, ch_out = 1):
		x, _, _ = common.create_conv_layer('cnn_l1', x, ch_in, 128, kx = 5, ky = 5, stride_x = 1, stride_y = 1, is_training = self.is_training, batchnorm = True)
		x, _, _ = common.create_conv_layer('cnn_l2', x, 128, 128, kx = 3, ky = 3, stride_x = 1, stride_y = 1, is_training = self.is_training, batchnorm = True)
		x, _, _ = common.create_conv_layer('cnn_l3', x, 128, ch_out, kx = 3, ky = 3, stride_x = 1, stride_y = 1, is_training = self.is_training, batchnorm = False, activation="linear")
		
		return x 

	def buildUNET_Regular(self, x, ch_in = 16, ch = 32, ch_out = 2):

		x, _, _ = common.create_conv_layer('cnn_l1', x, ch_in, ch*2, kx = 3, ky = 3, stride_x = 1, stride_y = 1, is_training = self.is_training, batchnorm = True)
		x1 = x 
		x, _, _ = common.create_conv_layer('cnn_l2', x, ch*2, ch*4, kx = 3, ky = 3, stride_x = 2, stride_y = 2, is_training = self.is_training, batchnorm = True)
		
		x, _, _ = common.create_conv_layer('cnn_l3', x, ch*4, ch*4, kx = 3, ky = 3, stride_x = 1, stride_y = 1, is_training = self.is_training, batchnorm = True)
		x2 = x 
		x, _, _ = common.create_conv_layer('cnn_l4', x, ch*4, ch*8, kx = 3, ky = 3, stride_x = 2, stride_y = 2, is_training = self.is_training, batchnorm = True)
		
		x, _, _ = common.create_conv_layer('cnn_l5', x, ch*8, ch*8, kx = 3, ky = 3, stride_x = 1, stride_y = 1, is_training = self.is_training, batchnorm = True)
		x, _, _ = common.create_conv_layer('cnn_l6', x, ch*8, ch*8, kx = 3, ky = 3, stride_x = 1, stride_y = 1, is_training = self.is_training, batchnorm = True)
		
		x, _, _ = common.create_conv_layer('cnn_l7', x, ch*8, ch*4, kx = 3, ky = 3, stride_x = 2, stride_y = 2, is_training = self.is_training, batchnorm = True, deconv = True)
		x = tf.concat([x, x2], axis=3)
		x, _, _ = common.create_conv_layer('cnn_l8', x, ch*4 + ch*4, ch*4, kx = 3, ky = 3, stride_x = 1, stride_y = 1, is_training = self.is_training, batchnorm = True)
		
		x, _, _ = common.create_conv_layer('cnn_l9', x, ch*4, ch*2, kx = 3, ky = 3, stride_x = 2, stride_y = 2, is_training = self.is_training, batchnorm = True, deconv = True)
		x = tf.concat([x, x1], axis=3)
		x, _, _ = common.create_conv_layer('cnn_l10', x, ch*2 + ch*2, ch*2, kx = 3, ky = 3, stride_x = 1, stride_y = 1, is_training = self.is_training, batchnorm = True)
		
		out, _, _ = common.create_conv_layer('cnn_out', x, ch*2, ch_out, kx = 3, ky = 3, stride_x = 1, stride_y = 1, is_training = self.is_training, batchnorm = False, activation="linear")

		return out

	def buildUNET_Tiny(self, x, ch_in = 16, ch = 32):

		x, _, _ = common.create_conv_layer('cnn_l1', x, ch_in, ch*2, kx = 3, ky = 3, stride_x = 1, stride_y = 1, is_training = self.is_training, batchnorm = True)
		x1 = x 
		x, _, _ = common.create_conv_layer('cnn_l2', x, ch*2, ch*2, kx = 3, ky = 3, stride_x = 2, stride_y = 2, is_training = self.is_training, batchnorm = True)
		
		x, _, _ = common.create_conv_layer('cnn_l3', x, ch*2, ch*2, kx = 3, ky = 3, stride_x = 1, stride_y = 1, is_training = self.is_training, batchnorm = True)
		x2 = x 
		x, _, _ = common.create_conv_layer('cnn_l4', x, ch*2, ch*4, kx = 3, ky = 3, stride_x = 2, stride_y = 2, is_training = self.is_training, batchnorm = True)
		
		x, _, _ = common.create_conv_layer('cnn_l5', x, ch*4, ch*4, kx = 3, ky = 3, stride_x = 1, stride_y = 1, is_training = self.is_training, batchnorm = True)
		x, _, _ = common.create_conv_layer('cnn_l6', x, ch*4, ch*4, kx = 3, ky = 3, stride_x = 1, stride_y = 1, is_training = self.is_training, batchnorm = True)
		
		x, _, _ = common.create_conv_layer('cnn_l7', x, ch*4, ch*2, kx = 3, ky = 3, stride_x = 2, stride_y = 2, is_training = self.is_training, batchnorm = True, deconv = True)
		x = tf.concat([x, x2], axis=3)
		x, _, _ = common.create_conv_layer('cnn_l8', x, ch*2 + ch*2, ch*2, kx = 3, ky = 3, stride_x = 1, stride_y = 1, is_training = self.is_training, batchnorm = True)
		
		x, _, _ = common.create_conv_layer('cnn_l9', x, ch*2, ch*2, kx = 3, ky = 3, stride_x = 2, stride_y = 2, is_training = self.is_training, batchnorm = True, deconv = True)
		x = tf.concat([x, x1], axis=3)
		x, _, _ = common.create_conv_layer('cnn_l10', x, ch*2 + ch*2, ch*2, kx = 3, ky = 3, stride_x = 1, stride_y = 1, is_training = self.is_training, batchnorm = True)
		
		out, _, _ = common.create_conv_layer('cnn_out', x, ch*2, 1, kx = 3, ky = 3, stride_x = 1, stride_y = 1, is_training = self.is_training, batchnorm = False, activation="linear")

		return out

	def buildUNET_resnet50(self, x, ch_in = 16, ch = 32, ch_out = 1):
		for i in range(3) :
			x = residual_block(x, channels=ch, is_training=self.is_training, downsample=False, scope='resblock0_' + str(i))

		f1 = relu(batch_norm_resnet(x, self.is_training, scope='batch_norm_f1'))
		########################################################################################################
		
		x = residual_block(x, channels=ch*2, is_training=self.is_training, downsample=True, scope='resblock1_0')

		for i in range(1,4) :
			x = residual_block(x, channels=ch*2, is_training=self.is_training, downsample=False, scope='resblock1_' + str(i))
		f2 = relu(batch_norm_resnet(x, self.is_training, scope='batch_norm_f2')) 
		f2 = tf.image.resize(f2, [self.input_dim, self.input_dim])
		
		########################################################################################################
		
		x = residual_block(x, channels=ch*4, is_training=self.is_training, downsample=True, scope='resblock2_0')

		for i in range(1,6) :
			x = residual_block(x, channels=ch*4, is_training=self.is_training, downsample=False, scope='resblock2_' + str(i))

		f3 = relu(batch_norm_resnet(x, self.is_training, scope='batch_norm_f3')) 
		f3 = tf.image.resize(f3, [self.input_dim, self.input_dim])
		
		########################################################################################################

		x = residual_block(x, channels=ch*8, is_training=self.is_training, downsample=True, scope='resblock_3_0')

		for i in range(1,3) :
			x = residual_block(x, channels=ch*8, is_training=self.is_training, downsample=False, scope='resblock_3_' + str(i))

		f4 = relu(batch_norm_resnet(x, self.is_training, scope='batch_norm_f4'))
		f4 = tf.image.resize(f4, [self.input_dim, self.input_dim])
		########################################################################################################


		features = tf.concat([f1,f2,f3,f4], axis=3) # ch * 15

		x, _, _ = common.create_conv_layer('cnn_l1', features, ch*15, ch*15, kx = 1, ky = 1, stride_x = 1, stride_y = 1, is_training = self.is_training, batchnorm = True)
		x, _, _ = common.create_conv_layer('cnn_l2', features, ch*15, ch*15, kx = 1, ky = 1, stride_x = 1, stride_y = 1, is_training = self.is_training, batchnorm = True)
		x, _, _ = common.create_conv_layer('cnn_l3', features, ch*15, ch_out, kx = 1, ky = 1, stride_x = 1, stride_y = 1, is_training = self.is_training, batchnorm = False, activation="linear")
		
		return x 

	def buildUNET_resnet18(self, x, ch_in = 16, ch = 32, ch_out=1):
		print("resnet18 channel ", ch_in, ch_out, ch)
		for i in range(2) :
			x = residual_block(x, channels=ch, is_training=self.is_training, downsample=False, scope='resblock0_' + str(i))

		f1 = relu(batch_norm_resnet(x, self.is_training, scope='batch_norm_f1'))
		########################################################################################################
		
		x = residual_block(x, channels=ch*2, is_training=self.is_training, downsample=True, scope='resblock1_0')

		for i in range(1,2) :
			x = residual_block(x, channels=ch*2, is_training=self.is_training, downsample=False, scope='resblock1_' + str(i))
		f2 = relu(batch_norm_resnet(x, self.is_training, scope='batch_norm_f2')) 
		f2 = tf.image.resize(f2, [self.input_dim, self.input_dim])
		
		########################################################################################################
		
		x = residual_block(x, channels=ch*4, is_training=self.is_training, downsample=True, scope='resblock2_0')

		for i in range(1,2) :
			x = residual_block(x, channels=ch*4, is_training=self.is_training, downsample=False, scope='resblock2_' + str(i))

		f3 = relu(batch_norm_resnet(x, self.is_training, scope='batch_norm_f3')) 
		f3 = tf.image.resize(f3, [self.input_dim, self.input_dim])
		
		########################################################################################################

		x = residual_block(x, channels=ch*8, is_training=self.is_training, downsample=True, scope='resblock_3_0')

		for i in range(1,2) :
			x = residual_block(x, channels=ch*8, is_training=self.is_training, downsample=False, scope='resblock_3_' + str(i))

		f4 = relu(batch_norm_resnet(x, self.is_training, scope='batch_norm_f4'))
		f4 = tf.image.resize(f4, [self.input_dim, self.input_dim])
		########################################################################################################


		features = tf.concat([f1,f2,f3,f4], axis=3) # ch * 15

		x, _, _ = common.create_conv_layer('cnn_l1', features, ch*15, ch*15, kx = 1, ky = 1, stride_x = 1, stride_y = 1, is_training = self.is_training, batchnorm = True)
		x, _, _ = common.create_conv_layer('cnn_l2', features, ch*15, ch*15, kx = 1, ky = 1, stride_x = 1, stride_y = 1, is_training = self.is_training, batchnorm = True)
		x, _, _ = common.create_conv_layer('cnn_l3', features, ch*15, ch_out, kx = 1, ky = 1, stride_x = 1, stride_y = 1, is_training = self.is_training, batchnorm = False, activation="linear")
		
		return x 
			


	def Train(self, target, sat, road, gps2d, gps_features, lr, history=None, accum = False):
		feed_dict = {
			self.target_in:target, 
			self.input_sat_in : sat,
			self.input_road_seg_in : road,
			self.input_gps_2d_in : gps2d,
			self.input_gps_features_in : gps_features,
			self.lr:lr,
			self.is_training:True
		}

		if USE_TEMPORAL_SPLIT:
			feed_dict[self.history_in] = history

		#ops = self.losses + self.outputs + self.trainops

		ops = self.losses + self.outputs
		ops += self.trainops

		if USE_TEMPORAL_SPLIT:
			ops += [self.gate, self.hist, self.pred]

		return self.sess.run(ops, feed_dict = feed_dict)

	
	def Evaluate(self, inputtraces, target, sat, road, gps2d, gps_features,gps_features_ds, history=None, st = None, ed = None):
		feed_dict = {
			self.target_in:target,
			self.input_sat_in : sat,
			self.input_road_seg_in : road,
			self.input_gps_2d_in : gps2d,
			self.input_gps_features_in : gps_features,
			self.is_training:False
		}

		if USE_TEMPORAL_SPLIT:
			feed_dict[self.history_in] = history
		if st is not None and ed is not None:
			ops = self.losses[st:ed] + self.outputs[st:ed]
		else:	
			ops = self.losses + self.outputs

		return self.sess.run(ops, feed_dict = feed_dict)
	
	
	def saveModel(self, path):
		self.saver.save(self.sess, path)

	def restoreModel(self, path):
		self.saver.restore(self.sess, path)

	def addLog(self, test_loss, train_loss):
		feed_dict = {
			self.test_loss : test_loss,
			self.train_loss : train_loss,
		}
		return self.sess.run(self.merged_summary, feed_dict=feed_dict)



