import re
import numpy as np
from preprocessing.replace import Replacer

class Tokenizer:
    def __init__(self, vocab_size = 1000):
        self.word_index = dict()
        self.index_word = dict()
        self.replacer = Replacer()
        self.num_words = dict()
        self.index = 1
        self.bag_of_words = []
        self.bag_of_onehot = []

        
        self.vocab_size = vocab_size

        self.word_one_hot_matrix = dict()
    def fix_config(self, word):
        if word not in self.word_index:
            self.word_index[word] = self.index
            self.num_words[word] = 1
            self.index_word[self.index] = word
            self.index += 1
            return True
        else:
            self.num_words[word] += 1
            return False

    def preprocessing(self, sequence, lang='en'):
        sequence = sequence.lower()
        if lang == 'en':
            sequence = self.replacer.replace(sequence)
        sequence = re.sub(r'[,.!@]', '', sequence)
        sequence = re.sub(r'\n', ' ', sequence)
        sequence = re.sub(r'\s\s+', ' ', sequence.strip())

        sequence = f"<bos> {sequence} <eos>"
        return sequence
    
    def fit_on_texts(self, sequences, lang='en'):
        for sequence in sequences:
            sequence = self.preprocessing(sequence, lang)
            words = sequence.split(' ')
            for word in words:
                self.fix_config(word)

    def texts_to_sequences(self, texts):
        output = []
        for text in texts:
            text = self.preprocessing(text)
            sequence_arr = []
            for word in text.split(' '):
                self.fix_config(word)
                sequence_arr.append(self.word_index[word])
            output.append(sequence_arr)
        return output
    
    def pad_sequence(self, input_sequence, truncating = 'post', padding='post', max_length = 1000):
        for sequence in input_sequence:
            delta =  np.abs(len(sequence) - max_length)
            if len(sequence) >= max_length:
                if truncating == 'post':
                    sequence = sequence[0:max_length-1]
                else:
                    if delta < max_length:
                        sequence = sequence[(max_length-delta):len(sequence)]
                    else:
                        sequence = sequence[delta:len(sequence)]
            else:
                if padding == 'post':
                    for _ in range(delta):
                        sequence.append(0)
                else:
                    temp = [0 for _ in range(max_length)]

                    for i in range(delta):
                        temp[delta + i] = sequence[i]
                    sequence = temp
            sequence = np.array(sequence)

        return np.array(input_sequence)

    def fit_to_bag(self, sequences, window_size = 2):
        for sequence in sequences:
            sequence = self.preprocessing(sequence)
            word_arr = sequence.split(' ')
            len_words = len(word_arr)
            for index, word in enumerate(word_arr):
                self.fix_config(word)
                begin = index - window_size
                end = index + window_size + 1
                context = [word_arr[i] for i in range(begin, end) if 0 <= i < len_words and i != index]
                target = word
                self.bag_of_words.append((context, target))

    def one_hot_encoding(self):
        self.one_hot_matrix = np.zeros((self.index, self.index))
        for i in range(self.index):
            for j in range(self.index):
                if i == j and i != 0:
                    self.one_hot_matrix[i][j] = 1

    def find_one_hot_vector(self, word):
        if type(word) == str:
            index_word = self.word_index[word]
            return np.reshape(self.one_hot_matrix[index_word], (self.index, 1))
        else:
            return np.reshape(self.one_hot_matrix[word], (self.index, 1))
    
    