from keras.layers import Layer, LayerNormalization, Dense, Dropout
import tensorflow as tf
from transformer.layers.multi_head_attention import MultiHeadAttention
from transformer.behaviors.position_wise_feed_forward_networks import ffn

class DecoderLayer(Layer):
    def __init__(self, h, d_ff, d_model, activation, dropout_rate=0.1, eps=0.1):
        super(DecoderLayer, self).__init__()
        self.masked_multi_head_attention = MultiHeadAttention(d_model, h)
        self.multi_head_attention = MultiHeadAttention(d_model, h)
        self.ffn = ffn(d_ff, d_model, activation)

        self.norm_layer_1 = LayerNormalization(epsilon=eps)
        self.norm_layer_2 = LayerNormalization(epsilon=eps)
        self.norm_layer_3 = LayerNormalization(epsilon=eps)

        self.dropout_layer_1 = Dropout(dropout_rate)
        self.dropout_layer_2 = Dropout(dropout_rate)
        self.dropout_layer_3 = Dropout(dropout_rate)

    def call(self, tensor, encoder_out, is_train,  look_ahead_mask=None, padding_mask=None):
        masked_multi_head_attention_output, self_attention_output = self.masked_multi_head_attention(tensor, tensor, tensor, look_ahead_mask)
        
        tensor = self.norm_layer_1(tensor + self.dropout_layer_1(masked_multi_head_attention_output, training=is_train))

        k = v = encoder_out

        multi_head_attention_out, global_attention_output = self.multi_head_attention(tensor, k, v, padding_mask)

        tensor = self.norm_layer_2(tensor + self.dropout_layer_2(multi_head_attention_out, training=is_train))

        ffn_out = self.ffn(tensor)

        decoder_out = self.norm_layer_3(tensor + self.dropout_layer_3(ffn_out, training=is_train))

        return decoder_out, self_attention_output ,global_attention_output


        