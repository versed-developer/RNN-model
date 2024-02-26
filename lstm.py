import os
import shutil

import numpy as np
import pandas as pd
import tensorflow.keras as K
from tensorflow.keras.layers import Dense, LSTM, Concatenate, Input, Attention, Bidirectional
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.utils import to_categorical

phase_list = ['A', 'B', 'C']


def load_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(file_path)

    columns = [f"t{i}" for i in range(1, 1441)]
    data_full_train = pd.read_excel(file_path, sheet_name='Full', index_col=0, names=columns)
    g_truth_data = pd.read_excel(file_path, sheet_name='G truth')

    node_to_phase = {}
    for phase, nodes in g_truth_data.items():
        for node in nodes.dropna():
            node_to_phase[node] = phase_list.index(phase)

    data_full_train['PHASE'] = data_full_train.index.map(node_to_phase)

    x = np.expand_dims(data_full_train.drop(columns=['PHASE']).to_numpy(), axis=(2,))
    y = to_categorical(data_full_train['PHASE'].to_numpy(), num_classes=3)

    return x, y, data_full_train.index


def build_rnn_model():
    _input = Input(shape=(1440, 1))

    lstm_out = Bidirectional(LSTM(128))(_input)
    #lstm_out = Dropout(0.4)(lstm_out)

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


def remove_logs(path):
    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")


if __name__ == '__main__':
    remove_logs('logs')
    x_train, y_train, _ = load_data('training.xlsx')
    print(x_train.shape)
    print(y_train.shape)
    model = build_rnn_model()
    # model = build_dense_model()

    # Enable below code block if you need to save checkpoints
    # checkpoint_path = "./weights/training/cp-{epoch:04d}.ckpt"
    # checkpoint_dir = os.path.dirname(checkpoint_path)
    #
    # cp_callback = K.callbacks.ModelCheckpoint(
    #    checkpoint_path, verbose=1, save_weights_only=True, save_freq='epoch'
    # )

    tensorboard_callback = K.callbacks.TensorBoard(log_dir="./logs")
    model.fit(x_train, y_train, epochs=256, batch_size=16, callbacks=[tensorboard_callback])
    # model.fit(x_train, y_train, epochs=256, batch_size=16, callbacks=[tensorboard_callback, cp_callback])
    # model.fit(x_train, y_train, validation_split=0.1, epochs=256, batch_size=8, callbacks=[tensorboard_callback])
    # model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=256, batch_size=8, callbacks=[tensorboard_callback])

    # Final evaluation of the model using testing.xlsx
    x_test, y_test, _ = load_data('testing.xlsx')
    scores = model.evaluate(x_test, y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1]*100))

    save_model = input("Save current model?")
    if save_model == "y" or save_model == "Y":
        model.save('lstm_model.keras')
