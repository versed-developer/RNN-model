import numpy as np
import pandas as pd

import tensorflow.keras as K
from tensorflow.keras.layers import Dense, LSTM, Concatenate, Dropout, Input, Attention
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.utils import to_categorical


def load_data(file_path, include_g_truth=True):
    columns = [f"t{i}" for i in range(1, 1441)]
    data_full_train = pd.read_excel(file_path, sheet_name='Full', index_col=0, names=columns)
    if include_g_truth:
        g_truth_data = pd.read_excel(file_path, sheet_name='G truth')

        phase_list = ['A', 'B', 'C']
        node_to_phase = {}
        for phase, nodes in g_truth_data.items():
            for node in nodes.dropna():
                node_to_phase[node] = phase_list.index(phase)

        data_full_train['PHASE'] = data_full_train.index.map(node_to_phase)

    x = data_full_train.drop(columns=['PHASE']).to_numpy() if include_g_truth else data_full_train.to_numpy()
    x = np.expand_dims(x, axis=(2,))

    y = to_categorical(data_full_train['PHASE'].to_numpy(), num_classes=3) if include_g_truth else None

    return x, y


def build_rnn_model():
    _input = Input(shape=(1440, 1))

    lstm_out = LSTM(128)(_input)
    lstm_out = Dropout(0.4)(lstm_out)

    attention = Attention()([lstm_out, lstm_out])
    out = Concatenate()([lstm_out, attention])

    out = Dense(512, activation='relu')(out)
    out = Dense(256, activation='relu')(out)
    out = Dense(3, activation='softmax')(out)

    rnn_model = Model(inputs=_input, outputs=out)

    rnn_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(rnn_model.summary())

    return rnn_model


def build_dense_model():
    dense_model = Sequential()
    dense_model.add(Dense(512, input_shape=(1440, ), activation='relu'))
    dense_model.add(Dense(512, activation='relu'))
    dense_model.add(Dense(512, activation='relu'))
    dense_model.add(Dense(256, activation='relu'))
    #model.add(Dropout(0.4))
    dense_model.add(Dense(256, activation='relu'))
    dense_model.add(Dense(3, activation='softmax'))

    dense_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(dense_model.summary())

    return dense_model


x_train, y_train = load_data('training.xlsx')
print(x_train.shape)
print(y_train.shape)
model = build_rnn_model()
# model = build_dense_model()

tensorboard_callback = K.callbacks.TensorBoard(log_dir="./logs")
model.fit(x_train, y_train, epochs=256, batch_size=8, callbacks=[tensorboard_callback])
# model.fit(x_train, y_train, validation_split=0.1, epochs=256, batch_size=8, callbacks=[tensorboard_callback])
# model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=256, batch_size=8, callbacks=[tensorboard_callback])

# Final evaluation of the model
x_test, y_test = load_data('training.xlsx')
scores = model.evaluate(x_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
