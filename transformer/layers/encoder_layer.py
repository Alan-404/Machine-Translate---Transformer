from keras.layers import Layer, LayerNormalization, Dropout
import tensorflow as tf
from transformer.layers.multi_head_attention import MultiHeadAttention
from transformer.behaviors.position_wise_feed_forward_networks import ffn

class EncoderLayer(Layer):
    def __init__(self, h, d_model, d_ff, activation, dropout_rate=0.1, eps=0.1):
        super(EncoderLayer, self).__init__()
        self.multi_head_attention = MultiHeadAttention(d_model, h)
        self.feed_forward = ffn(d_ff, d_model, activation)
        self.norm_layer_1 = LayerNormalization(epsilon=eps)
        self.norm_layer_2 = LayerNormalization(epsilon=eps)
        self.dropout_layer_1 = Dropout(dropout_rate)
        self.dropout_layer_2 = Dropout(dropout_rate)

    def call(self, tensor, is_train, mask=None):

        multi_head_attention_out, _ = self.multi_head_attention(tensor, tensor, tensor, mask)

        tensor = self.norm_layer_1(tensor + self.dropout_layer_1(multi_head_attention_out, training=is_train))

        ffn_out = self.feed_forward(tensor)

        encoder_layer_out = self.norm_layer_2(tensor + self.dropout_layer_2(ffn_out, training=is_train))

        return encoder_layer_out

