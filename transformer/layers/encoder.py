from keras.layers import Layer, Dropout, Embedding
from transformer.behaviors.positional_encoding import encode_position
import tensorflow as tf
from transformer.layers.encoder_layer import EncoderLayer

class Encoder(Layer):
    def __init__(self, n, h, d_ff, d_model, vocab_size, activation,  dropout_rate=0.1, eps=0.1):
        super(Encoder, self).__init__()
        self.n = n
        self.d_model = d_model
        self.encoder_layers = [EncoderLayer(h, d_model, d_ff, activation, dropout_rate, eps) for _ in range(n)]
        self.embedded_layer = Embedding(input_dim=vocab_size, output_dim=d_model)
        self.dropout = Dropout(dropout_rate)

    def call(self, tensor, is_train, mask=None):
        length = tensor.shape[1]

        embedded = self.embedded_layer(tensor)

        embedded *= tf.math.sqrt(tf.cast(self.d_model, dtype=tf.float32))

        encoder_output = self.dropout(embedded + encode_position(length, self.d_model), training=is_train)

        for encoder_layer in self.encoder_layers:
            encoder_output = encoder_layer(encoder_output, is_train, mask)
        return encoder_output


