import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D
from sklearn.model_selection import train_test_split
import numpy as np

# Sample dataset
sentences = ['I love this product!', 'This is awful.', 'Best experience ever.', 'Worst purchase.']
labels = [1, 0, 1, 0]  # Positive = 1, Negative = 0

# Tokenization and padding
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(sentences)
X = tokenizer.texts_to_sequences(sentences)
X = pad_sequences(X, maxlen=100)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2)

# Build the model
model = Sequential()
model.add(Embedding(input_dim=5000, output_dim=64, input_length=100))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(np.array(X_train), np.array(y_train), epochs=5, batch_size=64, validation_data=(np.array(X_test), np.array(y_test)))
