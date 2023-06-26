from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from data_preprocessing import tokenizer, embedding_dim, max_sequence_length, X_train_tokens_padded, y_train, \
    X_test_tokens_padded, y_test

# model creation, compilation and learning
model = Sequential()
model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=embedding_dim, input_length=max_sequence_length))
model.add(LSTM(units=128))
model.add(Dense(units=6, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train_tokens_padded, y_train, validation_data=(X_test_tokens_padded, y_test), epochs=2, batch_size=32)

model.save("rnn_based_model_v2.h5")
