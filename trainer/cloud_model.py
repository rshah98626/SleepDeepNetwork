import keras as keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv1D
from keras.layers import Dropout, MaxPool1D
import keras.backend as K
from keras import callbacks
import tensorflow as tf
K.set_image_data_format('channels_last')
import numpy as np
import argparse
from tensorflow.python.lib.io import file_io
from sklearn.model_selection import train_test_split
import random
from sklearn.preprocessing import LabelBinarizer
import pandas as pd
import EDFFileReader


class Model:
    def __init__(self, nb_classes, se):
        # input dim
        in_shape = (3000, 1)

        # create model
        self.m = Sequential()
        self.m.add(Conv1D(64, 5, strides=3, activation='relu', input_shape=in_shape))
        self.m.add(Conv1D(128, 5, strides=1, activation='relu'))
        self.m.add(MaxPool1D(pool_size=2, strides=2))
        self.m.add(Dropout(0.2, seed=se))
        self.m.add(Conv1D(128, 13, strides=1, activation='relu'))
        self.m.add(Conv1D(256, 7, strides=1, activation='relu'))
        self.m.add(MaxPool1D(pool_size=2, strides=2))
        self.m.add(Conv1D(256, 7, strides=1, activation='relu'))
        self.m.add(Conv1D(64, 4, strides=1, activation='relu'))
        self.m.add(MaxPool1D(pool_size=2, strides=2))
        self.m.add(Conv1D(32, 3, strides=1, activation='relu'))
        self.m.add(Conv1D(64, 6, strides=1, activation='relu'))
        self.m.add(MaxPool1D(pool_size=2, strides=2))
        self.m.add(Conv1D(8, 5, strides=1, activation='relu'))
        self.m.add(Conv1D(8, 2, strides=1, activation='relu'))
        self.m.add(MaxPool1D(pool_size=2, strides=2))
        self.m.add(Flatten())
        self.m.add(Dense(64, activation='relu'))  # TODO paper mentions drop = 0.2 so check if that means another dropout layer
        self.m.add(Dense(nb_classes, activation='softmax'))

        self.m.compile(loss=keras.losses.categorical_crossentropy,
                       optimizer=keras.optimizers.Adam(lr=.0001, decay=.003), metrics=['accuracy'])


def get_data(class_num):
    all_signals, all_labels = EDFFileReader.read_all()
    fpz = all_signals[:, 0].reshape(-1, 3000, 1)
    eog = all_signals[:, 1].reshape(-1, 3000, 1)
    both = np.add(fpz, eog)

    both = (both - np.mean(both, axis=0)) / np.std(both, axis=0)
    both = both.reshape(-1, 3000, 1)

    return all_labels, [eog, fpz, both]


# def temp_get_data(class_num):
#     file_path_data = 'firstData.csv'
#     file_path_label = 'firstData_Label.csv'
#     data_csv = pd.read_csv(file_path_data, delimiter=';')
#     labels_csv = pd.read_csv(file_path_label, delimiter=';')
#
#     labels = np.array(labels_csv['Hypnogram'][:-1])
#     eog_in_data = np.array(data_csv['EOG horizontal[uV]'])
#     fpz_in_data = np.array(data_csv['EEG Fpz-Cz[uV]'])
#
#     return labels, eog_in_data, fpz_in_data


def main(job_dir, **args):
    # Setting up the path for saving logs
    logs_path = job_dir + 'logs/tensorboard/'
    class_num = 6

    with tf.device('/device:GPU:0'):
        se = 42
        random.seed(se)

        # parse data
        labels, input_data = get_data(class_num)

        for ind, dSet in enumerate(input_data):
            (trainX, testX, trainY, testY) = train_test_split(dSet, labels, test_size=0.3, random_state=se)
            (valX, testX, valY, testY) = train_test_split(testX, testY, test_size=0.5, random_state=se)

            # one hot encoding
            lb = LabelBinarizer()
            trainY = lb.fit_transform(trainY)
            testY = lb.transform(testY)
            valY = lb.transform(valY)

            # set training params (TODO maybe vary eventually)
            nb_classes = trainY.shape[1]
            epochs = 100
            batch_size = 128  # alt values: (32, 64, 128, 256)

            # create & train model
            NN = Model(nb_classes, se)
            tensorboard = callbacks.TensorBoard(log_dir=logs_path, histogram_freq=10, write_graph=True,
                                                write_images=True)
            NN.m.fit(trainX, trainY, callbacks=[tensorboard], batch_size=batch_size, epochs=epochs, shuffle=True,
                     verbose=1, validation_data=(valX, valY))

            # evaluate model
            NN.m.evaluate(testX, testY, verbose=1)

            model_name = 'model' + str(class_num)
            if ind == 0:
                model_name += 'eog'
            elif ind == 1:
                model_name += 'fpz'
            else:
                model_name += 'both'
            model_name += '.h5'

            NN.m.save(model_name)
            with file_io.FileIO(model_name, mode='r') as input_f:
                with file_io.FileIO(job_dir + 'model/' + model_name, mode='w+') as output_f:
                    output_f.write(input_f.read())


# App Runner
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Input Args
    parser.add_argument(
        '--job-dir',
        help='GCS location to write checkpoints and export models',
        required=True
    )
    '''
    parser.add_argument(
        '--class-num',
        help='Which label set is wanted',
        required=True
    )
    '''
    args = parser.parse_args()
    arguments = args.__dict__

    main(**arguments)
