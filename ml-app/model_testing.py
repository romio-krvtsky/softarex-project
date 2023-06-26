import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

model_name = "rnn_based_model_with_glove.h5"
model = load_model(model_name)

test_data = pd.read_csv("data/cleaned_test_data.csv")

X_test = test_data["comment_text"]
y_test = test_data[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]]


tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_test)

X_test_tokens = tokenizer.texts_to_sequences(X_test)

max_sequence_length = 100

X_test_tokens_padded = pad_sequences(X_test_tokens, maxlen=max_sequence_length, padding="post", truncating="post")

y_pred = model.predict(X_test_tokens_padded)
y_pred_binary = (y_pred > 0.5).astype(int)


accuracy = accuracy_score(y_test, y_pred_binary)
precision = precision_score(y_test, y_pred_binary, average='micro')
recall = recall_score(y_test, y_pred_binary, average='micro')
f1 = f1_score(y_test, y_pred_binary, average='micro')


print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
confusion_mat = confusion_matrix(y_test.values.argmax(axis=1), y_pred_binary.argmax(axis=1))
print("Confusion Matrix:")
print(confusion_mat)
