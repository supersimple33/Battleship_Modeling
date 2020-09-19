import tensorflow as tf

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