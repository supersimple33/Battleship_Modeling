import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, LeakyReLU, Add, Flatten, Dense

NUM_GAMES = 100000
FILTERS = 64 # 64 because its cool

def convLayerCluster(inp):
	m = Conv2D(filters=FILTERS,kernel_size=(3,3),padding="same",data_format="channels_first",use_bias=False,activation='linear',kernel_regularizer=reg)(inp)
	m = BatchNormalization(axis=1)(m)
	return LeakyReLU()(m)

def residualLayerCluster(inp):
	m = convLayerCluster(inp)
	m = Conv2D(filters=FILTERS,kernel_size=(3,3),padding="same",data_format="channels_first",use_bias=False,activation='linear',kernel_regularizer=reg)(m)
	m = BatchNormalization(axis=1)(m)
	m = Add()([inp,m])
	return LeakyReLU()(m)

reg = tf.keras.regularizers.L2(l2=0.0001)
inputLay = tf.keras.Input(shape=(12,10,10))

m = convLayerCluster(inputLay)
m = residualLayerCluster(m)

m = residualLayerCluster(m)
m = residualLayerCluster(m)

m = Conv2D(filters=2,kernel_size=(1,1),padding="same",data_format="channels_first",use_bias=False,activation='linear',kernel_regularizer=reg)(m)
m = BatchNormalization(axis=1)(m)
m = LeakyReLU()(m)

m = Flatten(data_format="channels_first")(m)
m = Dense(100,activation='sigmoid')(m)

model = tf.keras.Model(inputs=inputLay, outputs=m)

print(model.summary())



# for epoch in range(0,NUM_GAMES):
# 	pass
