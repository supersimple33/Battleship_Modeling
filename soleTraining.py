import tensorflow as tf
from tensorflow.keras.layers import InputLayer, Flatten, Dense, Dropout
import tensorflow.keras.backend as K

from customs import customAccuracy

# one_tensor = tf.constant(1.0)
# zero_tensor = tf.constant(0.0)
# @tf.function

# GET DATA
spec = (tf.TensorSpec(shape=(10, 10, 6), dtype=tf.int32, name=None), tf.TensorSpec(shape=(100,), dtype=tf.float32, name=None))
dataset = tf.data.experimental.load('saved_data',spec)
dataset = dataset.batch(32)

# SET LAYERS
model = tf.keras.Sequential()
model.add(InputLayer(input_shape=(10,10,6)))
model.add(Flatten())
model.add(Dense(150,activation='relu'))
model.add(Dense(100,activation='sigmoid'))

optim = tf.keras.optimizers.Adam(learning_rate=0.1)
error = tf.keras.metrics.MeanAbsoluteError()
# loss = tf.nn.sparse_softmax_cross_entropy_with_logits
# binary_crossentropy
model.compile(optimizer=optim,loss='binary_crossentropy',metrics=[error, customAccuracy])
with tf.device('/cpu:0'):
    print(f"Starting training with {len(dataset)} batches")
    model.fit(x=dataset, epochs=10, verbose=2, use_multiprocessing=True)
    # model.train_on_batch(x=dataset)
model.save('saved_model/my_model')
print("Model Saved")