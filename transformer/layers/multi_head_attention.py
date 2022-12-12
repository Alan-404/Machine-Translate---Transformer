import tensorflow as tf
from keras.layers import Layer, Dense


class MultiHeadAttention(Layer):
    def __init__(self, d_model=512, h=8):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.h = h

        self.dense_k = Dense(d_model)
        self.dense_q = Dense(d_model)
        self.dense_v = Dense(d_model)

        self.dense_output = Dense(d_model)

    def scaled_dot_product_attention(self, q, k, v, mask=None):
        """ 
            q: query
            k: key
            v: value
        """

        dk = tf.cast(tf.shape(k)[-1], dtype=tf.float32)



        attention_scores = tf.matmul(q, k, transpose_b=True)/tf.math.sqrt(dk)

        if mask is not None:
            attention_scores += (mask*(-1e30))

        attention_weights = tf.nn.softmax(attention_scores, axis=-1)

        

        output = tf.matmul(attention_weights, v)

        return output, attention_weights

    def splitting_head(self, tensor):
        batch_size = tf.shape(tensor)[0]
        length = tf.shape(tensor)[1]
        d_model = tf.shape(tensor)[2]


        heading_value = d_model // self.h

        tensor = tf.reshape(tensor, (batch_size, length, self.h, heading_value)) #

        tensor_head = tf.transpose(tensor, [0,2,1,3]) # (batch_size, self.h, length, heading_value)

        return tensor_head

    def call(self, q, k, v, mask=None):

        batch_size = tf.shape(q)[0]

        qw = self.dense_q(q)
        kw = self.dense_k(k)
        vw = self.dense_v(v)



        heading_q = self.splitting_head(qw)
        heading_k = self.splitting_head(kw)
        heading_v = self.splitting_head(vw)

        output, attention_weights = self.scaled_dot_product_attention(heading_q, heading_k, heading_v, mask)

        output = tf.transpose(output, [0,2,1,3])

        output = tf.reshape(output, (batch_size, tf.shape(qw)[1], self.d_model))

        output = self.dense_output(output)


        return output, attention_weights

