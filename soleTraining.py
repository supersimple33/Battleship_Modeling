import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import InputLayer, Flatten, Dense, Dropout, GlobalMaxPool1D, Add, Reshape, LocallyConnected1D, Activation
import tensorflow.keras.backend as K

import tensorflow_datasets as tfds

import pandas as pd

from customs import customAccuracy, buildModel

# one_tensor = tf.constant(1.0)
# zero_tensor = tf.constant(0.0)
# @tf.function

# GET DATA
spec = (tf.TensorSpec(shape=(10, 10, 6), dtype=tf.int32, name=None), tf.TensorSpec(shape=(100,), dtype=tf.float32, name=None))
dataset = tf.data.experimental.load('saved_data',spec)
# dataset = dataset.batch(32)
# dataset = dataset.take(1000)
# dataset = dataset.repeat(2)

# npIter = tfds.as_numpy(dataset)

def map_fn(data):
    return tf.unstack(data, axis=0)
dataset.element_spec

# SET LAYERS

model = buildModel()

testMat = np.zeros((10,10,6))
testMat[0][0][0] = 1
testMat[0][0][1] = 1
testMat[0][1][1] = 1
ten = tf.convert_to_tensor([testMat])
print(model(ten))

optim = tf.keras.optimizers.Adam(learning_rate=0.1)
error = tf.keras.metrics.MeanAbsoluteError()
# loss = tf.nn.sparse_softmax_cross_entropy_with_logits
# binary_crossentropy
model.compile(optimizer=optim,loss='binary_crossentropy',metrics=[error, customAccuracy])
with tf.device('/cpu:0'):
    print(f"Starting training with {len(dataset)} batches")
    model.fit(x=dataset, epochs=10, verbose=2, use_multiprocessing=True, shuffle=True)
    # model.fit(x=dataset, epochs=100, verbose=2, use_multiprocessing=True, steps_per_epoch=200, shuffle=True)
    # model.train_on_batch(x=dataset)
model.save('saved_model/test')
print("Model Saved")