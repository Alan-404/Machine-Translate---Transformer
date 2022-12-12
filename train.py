#%%
from preprocessing.process_data import Data
from transformer.module import Transformer
#%%
tokenizer_folder = "./tokenizers"
num_data = "all"
checkpoint_folder = './checkpoints'
# %%
data_processer = Data('en', 'vi', tokenizer_folder)
# %%
train = data_processer.build_dataset('./datasets/en_sents.txt', './datasets/vi_sents.txt', buffer_size=64, batch_size=64, max_length=40, num_data=num_data)
#%%
inp_tokenizer = data_processer.inp_tokenizer
targ_tokenizer = data_processer.targ_tokenizer
# %%
inp_vocab_size = len(inp_tokenizer.word_counts) + 1
targ_vocab_size = len(targ_tokenizer.word_counts) + 1
#%%
targ_vocab_size
# %%
model = Transformer(input_vocab_size=inp_vocab_size, target_vocab_size=targ_vocab_size, checkpoint_folder=checkpoint_folder)
#%%
my_model = model.fit(train, epochs=1, saved_checkpoint_at=1)
# %%
my_model = model.fit(train, epochs=5)
# %%
    
#%%
inp_data = "Be quiet"
#%%
inp_data = data_processer.preprocess_sentence(inp_data, 40)
# %%

# %%
sentences = inp_tokenizer.texts_to_sequences([inp_data])
#%%
sentences
# %%

# %%
from keras_preprocessing.sequence import pad_sequences
# %%
tensor = pad_sequences(sentences, padding='post', maxlen=40)
#%%

# %%
import tensorflow as tf
encoder_input = tf.convert_to_tensor(tensor, dtype=tf.int64)
# %%
start, end = targ_tokenizer.word_index["<bos>"], targ_tokenizer.word_index["<eos>"]
# %%

# %%
targ_tokenizer.word_index
# %%
decoder_input = tf.convert_to_tensor([start], dtype=tf.int64)
decoder_input = tf.expand_dims(decoder_input, 0)
# %%
result = model.predict(encoder_input, decoder_input, False, 100, end, start)
# %%
final = targ_tokenizer.sequences_to_texts(result.numpy().tolist())
print('---------> result: ', " ".join(final[0].split()[1:]))
# %%
