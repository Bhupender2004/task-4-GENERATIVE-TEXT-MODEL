
import tensorflow as tf
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# Load Dataset
corpus = [
    "Artificial Intelligence is a field of computer science that focuses on creating intelligent machines.",
    "Machine Learning is a subset of AI that involves training algorithms to make predictions.",
    "Deep Learning is a specialized branch of machine learning based on neural networks.",
    "Natural Language Processing allows machines to understand and generate human language.",
    "Generative models can produce coherent text by learning patterns from input data."
]

# Preprocessing Data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index) + 1

input_sequences = []
for line in corpus:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

max_sequence_length = max([len(seq) for seq in input_sequences])
input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_length, padding='pre')

X = input_sequences[:, :-1]
y = input_sequences[:, -1]
y = tf.keras.utils.to_categorical(y, num_classes=total_words)

# Building the Model
model = Sequential()
model.add(Embedding(total_words, 100, input_length=max_sequence_length - 1))
model.add(LSTM(150, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(100))
model.add(Dense(total_words, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Training the Model
history = model.fit(X, y, epochs=100, verbose=1)

# Text Generation
def generate_text(seed_text, next_words, max_sequence_length):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_length-1, padding='pre')
        predicted = model.predict(token_list, verbose=0)
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == np.argmax(predicted):
                output_word = word
                break
        seed_text += " " + output_word
    return seed_text

# Example Usage
print(generate_text("Artificial Intelligence", 10, max_sequence_length))
