from keras.models import Model
from transformer.layers.encoder import Encoder
from transformer.layers.decoder import Decoder
from keras.metrics import Mean
from keras.optimizers import Adam
import tensorflow as tf
from keras.layers import Dense

class TransformerModel(Model):
    def __init__(self, n=6, h=8, input_vocab_size=10000, target_vocab_size=10000, d_model=512, d_ff=2048, activation='relu', dropout_rate=0.1, eps=0.1):
        super(TransformerModel, self).__init__()
        self.encoder = Encoder(n, h, d_ff, d_model, input_vocab_size, activation, dropout_rate, eps)
        self.decoder = Decoder(n, h, d_ff, d_model, target_vocab_size, activation, dropout_rate, eps)
        self.ffn = Dense(target_vocab_size)
        self.optimizer = Adam(learning_rate=0.01)

        self.train_accuracy = Mean(name='train_accuracy')
        self.loss_accuracy = Mean(name='train_loss')
        
        self.checkpoint = tf.train.Checkpoint(model=self, optimizer = self.optimizer) 

    def call(self, encoder_in, decoder_in, is_train, encoder_padding_mask, decoder_look_ahead_mask, decoder_padding_mask):
        encoder_output = self.encoder(encoder_in, is_train, encoder_padding_mask)

        decoder_output, _ = self.decoder(decoder_in, encoder_output, is_train, decoder_look_ahead_mask, decoder_padding_mask)
        return self.ffn(decoder_output)