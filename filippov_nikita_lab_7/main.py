import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.models import Sequential
from keras.utils import np_utils

SEQ_LENGTH = 100


def read_text():
    filename = "Prestuplenie_i_nakazanie.txt"
    text = open(filename, 'r', encoding='utf-8').read()
    text = text.lower()
    return text


def split_data():
    m_x = []
    m_y = []
    for i in range(0, text_size - SEQ_LENGTH, 1):
        seq_in = raw_text[i:i + SEQ_LENGTH]
        seq_out = raw_text[i + SEQ_LENGTH]
        m_x.append([char_to_int[char] for char in seq_in])
        m_y.append(char_to_int[seq_out])
    return m_x, m_y


def create_model(x, y):
    filepath_weights = "weights-improvement={epoch:02d}-{loss:4f}.hdf5"
    checkpoint = ModelCheckpoint(filepath_weights, monitor='loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]

    m_model = Sequential()
    m_model.add(LSTM(256, input_shape=(x.shape[1], x.shape[2]), return_sequences=True))
    m_model.add(Dropout(0.2))
    m_model.add(LSTM(256))
    m_model.add(Dropout(0.2))
    m_model.add(Dense(y.shape[1], activation='softmax'))

    m_model.load_weights ("weights-improvement=10-1.913465.hdf5")
    m_model.compile(loss='categorical_crossentropy', optimizer='adam')
    # m_model.fit(x, y, epochs=100, batch_size=128, callbacks=callbacks_list, verbose=1)
    return m_model


raw_text = read_text()

chars = sorted(list(set(raw_text)))
char_to_int = dict((char, i) for i, char in enumerate(chars))
int_to_char = dict((i, char) for i, char in enumerate(chars))

vocab_size = len(chars)
text_size = len(raw_text)

x_data, y_data = split_data()

n_patterns = len(x_data)

X = np.reshape(x_data, (n_patterns, SEQ_LENGTH, 1))
X = X / float(vocab_size)
y_data = np_utils.to_categorical(y_data)

model = create_model(X, y_data)

for i in range(10):
    generation = ""
    start = np.random.randint(0, len(x_data) - 1)
    pattern = x_data[start]
    print("С - " + str(i) + " " + "*\"", ''.join(int_to_char[value] for value in pattern), "\"*")
    for j in range(40):
        x = np.reshape(pattern, (1, len(pattern), 1))
        x = x / float(vocab_size)
        predication = model.predict(x, verbose=0)
        index = np.argmax(predication)
        result = int_to_char[index]
        seq_in = [int_to_char[value] for value in pattern]
        generation = generation + result
        pattern.append(index)
        pattern = pattern[1:len(pattern)]
    print("Г - " + generation)
