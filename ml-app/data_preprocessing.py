import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import gensim.downloader as api

train_data = pd.read_csv("data/train.csv")
test_data = pd.read_csv("data/test.csv")

X_train, X_test, y_train, y_test = train_test_split(train_data["comment_text"], train_data[
    ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]], test_size=0.2, random_state=42)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)

X_train_tokens = tokenizer.texts_to_sequences(X_train)
X_test_tokens = tokenizer.texts_to_sequences(X_test)

max_sequence_length = 100

X_train_tokens_padded = pad_sequences(X_train_tokens, maxlen=max_sequence_length, padding="post", truncating="post")
X_test_tokens_padded = pad_sequences(X_test_tokens, maxlen=max_sequence_length, padding="post", truncating="post")

embedding_dim = 100
word2vec_model = api.load("glove-wiki-gigaword-100")

embedding_matrix = np.zeros((len(tokenizer.word_index) + 1, embedding_dim))
for word, i in tokenizer.word_index.items():
    if word in word2vec_model:
        embedding_matrix[i] = word2vec_model[word]
