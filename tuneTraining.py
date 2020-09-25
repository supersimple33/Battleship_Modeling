import numpy as np
import os
# os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
# os.environ["RUNFILES_DIR"] = "/usr/local/share/plaidml"
# os.environ["PLAIDML_NATIVE_PATH"] = "/usr/local/lib/libplaidml.dylib"
import tensorflow as tf
print(tf.config.experimental.list_physical_devices())

tf.keras.backend.set_image_data_format('channels_first') #GPUs/TPUs

import tensorflow_datasets as tfds

import kerastuner as kt # for now should try others later

import pandas as pd

from customs import customAccuracy, buildModel
USING_TPU = False

if USING_TPU:
	resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='grpc://' + os.environ['COLAB_TPU_ADDR'])
	tf.config.experimental_connect_to_cluster(resolver)
	tf.tpu.experimental.initialize_tpu_system(resolver)
	print("All devices: ", tf.config.list_logical_devices('TPU'))
	strategy = tf.distribute.TPUStrategy(resolver)
elif tf.config.list_physical_devices('gpu'):
	strategy = tf.distribute.MirroredStrategy()
else:
	strategy = tf.distribute.get_strategy()

# GET DATA
spec = (tf.TensorSpec(shape=(6, 10, 10), dtype=tf.int32, name=None), tf.TensorSpec(shape=(100,), dtype=tf.float32, name=None))
# dataset = tf.data.experimental.load('saved_data',spec,compression='GZIP')

with open('data.npz', 'rb') as f:
	l = np.load(f)
# 	nA = tf.convert_to_tensor(l['arr_0'])
# 	nB = tf.convert_to_tensor(l['arr_1'])
	nA = l['arr_0'] #tf.stack(
	nB = l['arr_1']
	l.close()
# dA = tf.data.Dataset.from_tensor_slices(nA)
dataset = tf.data.Dataset.from_tensor_slices((nA,nB))

dataset.shuffle(1000, seed=tf.constant(0, dtype=tf.int64))
dataset = dataset.batch(32)
trainDataset = dataset.take(7500)
validationData = dataset.skip(7501)
validationData = validationData.take(750)
dataset = dataset.repeat()

# npIter = tfds.as_numpy(dataset)

def map_fn(data):
	return tf.unstack(data, axis=0)

# CREATE THE MODEL & TUNER
# trained_model = tf.keras.models.load_model('saved_model/test',compile=True,custom_objects={'customAccuracy':customAccuracy})
# ret = trained_model.fit(trainDataset, validation_data=validationData, epochs=10, shuffle=True, use_multiprocessing=True, workers=4,verbose=2)

def custModel(hp):
	# with strategy.scope():
	error = tf.keras.metrics.MeanAbsoluteError()
	model = buildModel(hp)
	lr = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
	optim = tf.keras.optimizers.Adam(learning_rate=lr)
	# model.compile(optimizer=optim,loss='binary_crossentropy',metrics=[error, customAccuracy],experimental_steps_per_execution = 50)
	model.compile(optimizer=optim,loss='binary_crossentropy',metrics=[error, customAccuracy])
	return model

#should be based on validation accuracy
obj = kt.Objective("val_customAccuracy", direction="max")
xmodel = kt.applications.xception.HyperXception(include_top=True, input_shape=(6,10,10), classes=100)
# tuner = kt.RandomSearch(custModel, objective=obj, max_trials=30,executions_per_trial=2,project_name="rstuner")
tuner = kt.Hyperband(xmodel, max_epochs=8, factor=2, hyperband_iterations=3, objective=obj, distribution_strategy=strategy, project_name="hbtuner", metrics=['mae', customAccuracy], loss='binary_crossentropy') #max_trials=40, max_epochs=6, executions_per_trial=2,

# with tf.device('CPU:2'):
# 	print(f"Starting training with {len(trainDataset)} batches")

tuner.search_space_summary()
tuner.search(trainDataset, validation_data=validationData, shuffle=True, use_multiprocessing=True, workers=4, verbose=2) #epochs
tuner.results_summary() #224,384,224,100 #0.001

model = tuner.get_best_models(num_models=2)[0]
model.save('saved_model/test')
print("Model Saved")