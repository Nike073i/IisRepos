import sys

import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.models import Sequential
from keras.utils import np_utils

filename = "Prestuplenie_i_nakazanie.txt"

raw_text = open(filename, 'r', encoding='utf-8').read()
raw_text = raw_text.lower()

chars = sorted(list(set(raw_text)))
char_to_int = dict((char, i) for i, char in enumerate(chars))
int_to_char = dict((i, char) for i, char in enumerate(chars))

vocab_size = len(chars)
text_size = len(raw_text)

seq_length = 100

x_data = []
y_data = []
for i in range(0, text_size - seq_length, 1):
    seq_in = raw_text[i:i + seq_length]
    seq_out = raw_text[i + seq_length]
    x_data.append([char_to_int[char] for char in seq_in])
    y_data.append(char_to_int[seq_out])

n_patterns = len(x_data)

X = np.reshape(x_data, (n_patterns, seq_length, 1))
X = X / float(vocab_size)

y_data = np_utils.to_categorical(y_data)

filepath_weights = "weights-improvement={epoch:02d}-{loss:4f}.hdf5"
checkpoint = ModelCheckpoint(filepath_weights, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256))
model.add(Dropout(0.2))
model.add(Dense(y_data.shape[1], activation='softmax'))

# model.load_weights ("")
model.compile(loss='categorical_crossentropy', optimizer='adam')
model.fit(X, y_data, epochs=100, batch_size=128, callbacks=callbacks_list, verbose=1)

start = np.random.randint(0, len(x_data) - 1)
pattern = x_data[start]

print("Seed:")
print("\"", ''.join(int_to_char[value] for value in pattern), "\"")
print("==========================")

for i in range(1000):
    x = np.reshape(pattern, (1, len(pattern), 1))
    x = x / float(vocab_size)
    predication = model.predict(x, verbose=0)
    index = np.argmax(predication)
    result = int_to_char[index]
    seq_in = [int_to_char[value] for value in pattern]
    sys.stdout.write(result)
    pattern.append(index)
    pattern = pattern[1:len(pattern)]
print("\nДон.")
