import tensorflow as tf
import numpy as np

def generate_padding_mask(inp):
    
    result = tf.cast(inp == 0, dtype=tf.float32)[:, np.newaxis, np.newaxis, :]
    
    return result # (batch_size, 1, 1, length_seq)


def generate_look_ahead_mask(inp_len):
    mask = 1 - tf.linalg.band_part(tf.ones((inp_len, inp_len)), -1, 0)
    return mask  # (length_seq, length_seq)

def generate_mask(inp, targ):

    encoder_padding_mask = generate_padding_mask(inp)
  
    decoder_padding_mask = generate_padding_mask(inp)
  
    decoder_look_ahead_mask = generate_look_ahead_mask(targ.shape[1])
  
    decoder_inp_padding_mask = generate_padding_mask(targ)

    decoder_look_ahead_mask = tf.math.maximum(decoder_look_ahead_mask, decoder_inp_padding_mask)

    return encoder_padding_mask, decoder_look_ahead_mask ,decoder_padding_mask