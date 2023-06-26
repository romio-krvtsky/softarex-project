import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


train_data = pd.read_csv("data/train.csv")

model_files = ["Bidirectional_LSTM_model.h5", "CNN_with_Global_Max_Pooling_model.h5", "LSTM_model.h5",
               "LSTM_with_Dropout_model.h5"]
models = []

for file in model_files:
    model = load_model(file)
    models.append(model)

text = input("Enter text: ")

tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_data["comment_text"])

text_tokens = tokenizer.texts_to_sequences([text])

max_sequence_length = 100

text_tokens_padded = pad_sequences(text_tokens, maxlen=max_sequence_length, padding="post", truncating="post")

for i, model in enumerate(models):
    prediction = model.predict(text_tokens_padded)
    prediction = (prediction > 0.5).astype(int)
    print(f"Предсказание модели {i + 1}: {prediction}")
