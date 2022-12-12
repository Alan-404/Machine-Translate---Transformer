import tensorflow as tf
from keras.losses import SparseCategoricalCrossentropy

loss_object = SparseCategoricalCrossentropy(from_logits=True, reduction='none')

def loss_function(loss_object, real, predict):
    mask = tf.math.logical_not(real == 0)
    loss = loss_object(real, predict)

    mask = tf.cast(mask, dtype=tf.float32)
    loss *= mask

    return tf.reduce_sum(loss)/tf.reduce_sum(mask)