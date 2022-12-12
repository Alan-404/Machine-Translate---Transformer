import tensorflow as tf
import numpy as np

def encode_position(length, d_model):
    angles = np.arange(d_model) # dim angles = (d_model, )
    angles[1::2] = angles[0::2] # 
    angles = 1/(10000**(angles/d_model)) # 

    angles = np.expand_dims(angles, axis=0) # dim angles = (1, d_model)

    length = np.expand_dims(np.arange(length), axis=1) # dim length = (length, 1)

    pos_angles = np.dot(length, angles) # dim pos_angles = (length, d_model)

    pos_angles[0::2] = np.sin(pos_angles[0::2])
    pos_angles[1::2] = np.cos(pos_angles[1::2])

    pos_angles = np.expand_dims(pos_angles, axis=0) # dim pos_angles = (1, length, d_model)

    return tf.cast(pos_angles, dtype=tf.float32)