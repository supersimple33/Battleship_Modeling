import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import InputLayer, Flatten, Dense, Dropout, GlobalMaxPool1D, Add, Reshape, LocallyConnected1D, Activation
import tensorflow.keras.backend as K

import tensorflow_datasets as tfds

import kerastuner as kt # for now should try others later

import pandas as pd

from customs import customAccuracy, buildModel

# one_tensor = tf.constant(1.0)
# zero_tensor = tf.constant(0.0)
# @tf.function

# GET DATA
spec = (tf.TensorSpec(shape=(10, 10, 6), dtype=tf.int32, name=None), tf.TensorSpec(shape=(100,), dtype=tf.float32, name=None))
dataset = tf.data.experimental.load('saved_data',spec)
dataset = dataset.batch(32)
trainDataset = dataset.take(12380//2)
validationData = dataset.skip(12380//2)
validationData = validationData.take(619)
# dataset = dataset.repeat(2)

# npIter = tfds.as_numpy(dataset)

def map_fn(data):
    return tf.unstack(data, axis=0)

# CREATE THE MODEL & TUNER
error = tf.keras.metrics.MeanAbsoluteError()

# trained_model = tf.keras.models.load_model('saved_model/test',compile=True,custom_objects={'customAccuracy':customAccuracy})
# ret = trained_model.fit(trainDataset, validation_data=validationData, epochs=10, shuffle=True, use_multiprocessing=True, workers=4,verbose=2)

def custModel(hp):
	model = buildModel(hp)
	lr = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
	optim = tf.keras.optimizers.Adam(learning_rate=lr)
	model.compile(optimizer=optim,loss='binary_crossentropy',metrics=[error, customAccuracy])
	return model

#should be based on validation accuracy
obj = kt.Objective("val_customAccuracy", direction="max")
# tuner = kt.RandomSearch(custModel, objective=obj, max_trials=20,executions_per_trial=2,project_name="tuner")
tuner = kt.Hyperband(custModel, factor=2, hyperband_iterations=4, objective=obj, max_epochs=6, executions_per_trial=2, project_name="hbtuner")

with tf.device('/cpu:0'):
	print(f"Starting training with {len(trainDataset)} batches")
	
	tuner.search_space_summary()
	tuner.search(trainDataset, validation_data=validationData, epochs=6, shuffle=True, use_multiprocessing=True, workers=4, verbose=2)
	tuner.results_summary() #224,384,224,100 #0.001

	model = tuner.get_best_models(num_models=2)[0]
model.save('saved_model/test')
print("Model Saved")