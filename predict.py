import pickle
from preprocessing.process_data import Data
from transformer.module import Transformer
from keras_preprocessing.sequence import pad_sequences
import tensorflow as tf
checkpoint_folder = "./checkpoints"
tokenizer_folder = './tokenizers'

class Predictor:
    def __init__(self):
        with open(f'./{tokenizer_folder}/en_tokenizer.pickle', 'rb') as handle:
            self.inp_tokenizer = pickle.load(handle)
        with open(f'./{tokenizer_folder}/vi_tokenizer.pickle', 'rb') as handle:
            self.targ_tokenizer = pickle.load(handle)
        self.inp_vocab_size = len(self.inp_tokenizer.word_counts) + 1
        self.targ_vocab_size = len(self.targ_tokenizer.word_counts) + 1
        self.data_processer = Data('en', 'vi', tokenizer_folder)
        self.model = Transformer(input_vocab_size=self.inp_vocab_size, target_vocab_size=self.targ_vocab_size, checkpoint_folder=checkpoint_folder)
    def post_handle(self, sequence):
        words = sequence.split(' ')
        result = []
        for index in range(len(words)):
            if words[index] == '<bos>':
                continue
            if index == 0:
                result.append(words[index])
            elif words[index] != words[index-1]:
                result.append(words[index])
        return " ".join(result)
    def predict(self, inp_data, maxlen=40):
        inp_data = self.data_processer.preprocess_sentence(inp_data, maxlen)
        sentence = self.inp_tokenizer.texts_to_sequences([inp_data])
        tensor = pad_sequences(sentence, padding='post', maxlen=maxlen)
        
        encoder_input = tf.convert_to_tensor(tensor, dtype=tf.int64)

        start, end = self.targ_tokenizer.word_index["<bos>"], self.targ_tokenizer.word_index["<eos>"]
        decoder_input = tf.convert_to_tensor([start], dtype=tf.int64)
        decoder_input = tf.expand_dims(decoder_input, 0)
        result = self.model.predict(encoder_input, decoder_input, False, maxlen, end, start)
        final = self.targ_tokenizer.sequences_to_texts(result.numpy().tolist())
        translated = self.post_handle(final[0])
        print('---------> result: ', translated)
        return translated