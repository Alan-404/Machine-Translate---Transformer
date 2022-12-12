from keras.layers import Layer, Dropout, Embedding
import tensorflow as tf
from transformer.layers.decoder_layer import DecoderLayer
from transformer.behaviors.positional_encoding import encode_position

class Decoder(Layer):
    def __init__(self,n, h, d_ff, d_model, vocab_size, activation,  dropout_rate=0.1, eps=0.1):
        super(Decoder, self).__init__()
        self.n = n
        self.d_model = d_model
        self.decoder_layers = [DecoderLayer(h, d_ff, d_model, activation, dropout_rate, eps) for _ in range(n)]

        self.embedded_seq = Embedding(input_dim=vocab_size, output_dim=d_model)
        self.dropout = Dropout(dropout_rate)

    def call(self, tensor, encoder_out, is_train, look_ahead_mask=None, padding_mask = None):
        length = tf.shape(tensor)[1]

        embedded = self.embedded_seq(tensor)

        position_encoded = encode_position(length, self.d_model)

        decoder_output = self.dropout(embedded + position_encoded, training=is_train)

        attention_weights = dict()

        for index, decoder_layer in enumerate(self.decoder_layers):
            decoder_output, self_attention_weights, global_attention_weights = decoder_layer(decoder_output, encoder_out, is_train, look_ahead_mask, padding_mask)
            attention_weights[f"decoder_layer_{index}_self_attention_weights"] = self_attention_weights
            attention_weights[f"decoder_layer_{index}_global_attention_weights"] = global_attention_weights

        return decoder_output, attention_weights


