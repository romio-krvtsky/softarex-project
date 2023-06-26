from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, Conv1D, GlobalMaxPooling1D, Dropout

from data_preprocessing import tokenizer, embedding_dim, max_sequence_length, embedding_matrix, X_train_tokens_padded, \
    y_train, X_test_tokens_padded, y_test

models = [
    # LSTM
    Sequential([
        Embedding(
            input_dim=len(tokenizer.word_index) + 1,
            output_dim=embedding_dim,
            input_length=max_sequence_length,
            weights=[embedding_matrix],
            trainable=False,
        ),
        LSTM(units=128),
        Dense(units=6, activation='sigmoid')
    ]),

    # Bidirectional LSTM
    Sequential([
        Embedding(
            input_dim=len(tokenizer.word_index) + 1,
            output_dim=embedding_dim,
            input_length=max_sequence_length,
            weights=[embedding_matrix],
            trainable=False,
        ),
        Bidirectional(LSTM(units=128)),
        Dense(units=6, activation='sigmoid')
    ]),

    # CNN with Global Max Pooling
    Sequential([
        Embedding(
            input_dim=len(tokenizer.word_index) + 1,
            output_dim=embedding_dim,
            input_length=max_sequence_length,
            weights=[embedding_matrix],
            trainable=False,
        ),
        Conv1D(filters=128, kernel_size=3, activation='relu'),
        GlobalMaxPooling1D(),
        Dense(units=6, activation='sigmoid')
    ]),

    # LSTM with Dropout
    Sequential([
        Embedding(
            input_dim=len(tokenizer.word_index) + 1,
            output_dim=embedding_dim,
            input_length=max_sequence_length,
            weights=[embedding_matrix],
            trainable=False,
        ),
        LSTM(units=128, dropout=0.2, recurrent_dropout=0.2),
        Dense(units=6, activation='sigmoid')
    ])
]

for i, model in enumerate(models):
    print(f"Training Model {i + 1}")
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train_tokens_padded, y_train, validation_data=(X_test_tokens_padded, y_test), epochs=2, batch_size=32)

# models[3].fit(X_train_tokens_padded, y_train, validation_data=(X_test_tokens_padded, y_test), epochs=5, batch_size=32)
# models[3].save("LSTM_with_Dropout_model.h5")
