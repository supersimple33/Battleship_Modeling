import tensorflow as tf
import tensorflow.keras.backend as K

@tf.function
def customAccuracy(y_true, y_pred):
    trans = tf.transpose([y_true, y_pred],perm=[1,0,2])
    accur = 0 
    i = 0
    for elem in trans:
        move = tf.argmax(elem[1],-1)
        v = elem[0][move]
        accur += 1 if v == 1 else 0
        i += 1
    return accur / i

@tf.function
def customLoss(y_true, y_pred):
	crossEnt = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true,logits=y_pred)
	return tf.reduce_mean(crossEnt)

@tf.function
def f1_loss(y_true, y_pred):
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
    return 1 - K.mean(f1)

def buildModel():
    inputLay = tf.keras.Input(shape=(10,10,6))
    f = Flatten()(inputLay)
    d1 = Dense(150,activation='relu')(f)
    d2 = Dense(100)(d1)

    r1 = Reshape((100,6))(inputLay)
    a = GlobalMaxPool1D('channels_first')(r1)
    r2 = Reshape((100,1))(a)
    lc = LocallyConnected1D(1,1)(r2)
    f2 = Flatten()(lc)
    # gm1 = GlobalMaxPool2D('channels_first')(inputLay)
    # gm2 = GlobalMaxPool1D('channels_first')(gm1)

    a = Add()([d2,f2])
    out = Activation('sigmoid')(a)
    return tf.keras.Model(inputs=inputLay, outputs=out)