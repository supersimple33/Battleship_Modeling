import numpy as np

import tensorflow as tf

import kerastuner as kt
tf.keras.backend.set_image_data_format('channels_last')

from customs import customAccuracy, buildModel, buildModel0

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

with open('data.npz', 'rb') as f:
	l = np.load(f)
	nA = l['arr_0'] #tf.stack(
	nB = l['arr_1']
	l.close()
dataset = tf.data.Dataset.from_tensor_slices((nA,nB))
dataset.shuffle(5000, seed=tf.constant(42, dtype=tf.int64)) # seed at 42 for consistent return
dataset = dataset.batch(32)

trainDataset = dataset.take(23035) #27100
validationData = dataset.skip(23035)

trainDataset = trainDataset.repeat()
validationData = validationData.repeat()

# obj = kt.Objective("val_customAccuracy", direction="max")
# xmodel = kt.applications.xception.HyperXception(include_top=True, input_shape=(6,10,10), classes=100)
# tuner = kt.Hyperband(xmodel, max_epochs=8, factor=2, hyperband_iterations=3, objective=obj, distribution_strategy=strategy, project_name="hbtuner", metrics=['mae', customAccuracy], loss='binary_crossentropy') #max_trials=40, max_epochs=6, executions_per_trial=2,

# model = tuner.get_best_models(num_models=1)[0]
# model.save('saved_model/test')

# model = tf.keras.models.load_model('saved_model/my_model',compile=True,custom_objects={'customAccuracy':customAccuracy})
model = buildModel0(None)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['mae', customAccuracy])

es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)
model.fit(trainDataset, validation_data=validationData, use_multiprocessing=True, workers=4, verbose=2, steps_per_epoch=22222, validation_steps=4065, callbacks=[es])
model.save('saved_model/test0')