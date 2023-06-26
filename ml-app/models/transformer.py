import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Attention, GlobalAveragePooling1D


data = pd.read_csv('C:\\pyProjects\\softarex-project\\data\\train.csv')

train_size = int(0.8 * len(data))
train_data = data[:train_size]
test_data = data[train_size:]


max_length = 100
tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_data['comment_text'])
train_sequences = tokenizer.texts_to_sequences(train_data['comment_text'])
test_sequences = tokenizer.texts_to_sequences(test_data['comment_text'])
train_padded = pad_sequences(train_sequences, maxlen=max_length)
test_padded = pad_sequences(test_sequences, maxlen=max_length)


embeddings_index = {}
with open('C:\pyProjects\softarex-project\glove\glove.6B.100d.txt', encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

embedding_dim = 100
vocab_size = len(tokenizer.word_index) + 1

embedding_matrix = np.zeros((vocab_size, embedding_dim))
for word, i in tokenizer.word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

model = Sequential()
model.add(Embedding(vocab_size, embedding_dim, weights=[embedding_matrix], input_length=max_length, trainable=False))
model.add(Attention())
model.add(GlobalAveragePooling1D())
model.add(Dense(6, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(train_padded, train_data[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']],
          validation_data=(test_padded, test_data[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']]),
          epochs=10, batch_size=32)

loss, accuracy = model.evaluate(test_padded, test_data[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']])
print('Test Loss:', loss)
print('Test Accuracy:', accuracy)
