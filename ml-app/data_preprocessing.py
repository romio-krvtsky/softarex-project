import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
import regex as re
from sklearn.model_selection import train_test_split
from gensim.models import KeyedVectors
import numpy as np

windows_data_path = "D:\\Softarex-project\\softarex-project\\ml-app\\dataset\\train\\train.csv"
linux_path = '../dataset/train/train.csv'
data = pd.read_csv(windows_data_path)


# Очистка текста
def clean_text(text):
    # Удаление лишних символов и преобразование в нижний регистр
    text = re.sub(r'[^a-zA-Zа-яА-Я0-9\s]', '', text)
    text = text.lower()
    return text


# Применение очистки текста к столбцу "comment_text"
data['comment_text'] = data['comment_text'].apply(clean_text)

train_data, valid_data, train_labels, valid_labels = train_test_split(data[['comment_text']], data[
    ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']], test_size=0.2, random_state=42)

train_tokenized = [word_tokenize(text) for text in train_data['comment_text']]
valid_tokenized = [word_tokenize(text) for text in valid_data['comment_text']]

embedding_dim = 100
windows_embedding_path = "D:\Softarex-project\glove\glove.6B.100d.txt"
embedding_file = '../../glove/glove.6B.300d.txt'

# Создание словаря для хранения эмбеддингов
embeddings_index = {}

# Загрузка предобученных эмбеддингов GloVe
with open(windows_embedding_path, 'r', encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

# Размер словаря (количество уникальных слов)
vocab_size = len(embeddings_index)

# Максимальная длина последовательности (максимальное количество слов в комментарии)
max_length = max(len(tokens) for tokens in train_tokenized)

# Преобразование токенизированных данных в числовые эмбеддинги
train_embeddings = []
valid_embeddings = []

for tokens in train_tokenized:
    token_embeddings = []
    for token in tokens:
        embedding = embeddings_index.get(token)
        if embedding is not None:
            token_embeddings.append(embedding)
    train_embeddings.append(token_embeddings)

for tokens in valid_tokenized:
    token_embeddings = []
    for token in tokens:
        embedding = embeddings_index.get(token)
        if embedding is not None:
            token_embeddings.append(embedding)
    valid_embeddings.append(token_embeddings)
