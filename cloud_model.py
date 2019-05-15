import keras as keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv1D
from keras.layers import Dropout, MaxPool1D
import keras.backend as K
import tensorflow as tf
# from keras import callbacks
from keras.callbacks import TensorBoard
import numpy as np
# import argparse
from sklearn.model_selection import train_test_split
import random
from sklearn.preprocessing import LabelBinarizer
import EDFFileReader
import sys
# import TrainValTensorBoard
import os
K.set_image_data_format('channels_last')


class TrainValTensorBoard(TensorBoard):
    def __init__(self, log_dir='./logs', **kwargs):
        # Make the original `TensorBoard` log to a subdirectory 'training'
        training_log_dir = os.path.join(log_dir, 'training')
        super(TrainValTensorBoard, self).__init__(training_log_dir, **kwargs)

        # Log the validation metrics to a separate subdirectory
        self.val_log_dir = os.path.join(log_dir, 'validation')

    def set_model(self, model):
        # Setup writer for validation metrics
        self.val_writer = tf.summary.FileWriter(self.val_log_dir)
        super(TrainValTensorBoard, self).set_model(model)

    def on_epoch_end(self, epoch, logs=None):
        # Pop the validation logs and handle them separately with
        # `self.val_writer`. Also rename the keys so that they can
        # be plotted on the same figure with the training metrics
        logs = logs or {}
        val_logs = {k.replace('val_', ''): v for k, v in logs.items() if k.startswith('val_')}
        for name, value in val_logs.items():
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.val_writer.add_summary(summary, epoch)
        self.val_writer.flush()

        # Pass the remaining logs to `TensorBoard.on_epoch_end`
        logs = {k: v for k, v in logs.items() if not k.startswith('val_')}
        super(TrainValTensorBoard, self).on_epoch_end(epoch, logs)

    def on_train_end(self, logs=None):
        super(TrainValTensorBoard, self).on_train_end(logs)
        self.val_writer.close()


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


def get_data(class_num, in_data_type):
    # TODO only using edf data as of now
    # select correct input data
    if in_data_type == 'fpz':
        # signals = np.concatenate((np.load('np_files/edf_fpz.npy'), np.load('np_files/edfx_fpz.npy')), axis=0)
        signals = np.load('np_files/edf_fpz.npy')
    elif in_data_type == 'eog':
        # signals = np.concatenate((np.load('np_files/edf_eog.npy'), np.load('np_files/edfx_eog.npy')), axis=0)
        signals = np.load('np_files/edf_eog.npy')
    elif in_data_type == 'both':
        # signals = np.concatenate((np.load('np_files/edf_both.npy'), np.load('np_files/edfx_both.npy')), axis=0)
        signals = np.load('np_files/edf_both.npy')
    else:
        raise Exception('Invalid third argument. Should be either fpz, eog, or both.')

    # clean label set
    # labels = np.concatenate((np.load('np_files/edf_labels.npy'), np.load('np_files/edfx_labels.npy')), axis=0)
    labels = np.load('np_files/edf_labels.npy')
    if class_num == 2:
        labels = EDFFileReader.create_class_two(labels)
    elif class_num == 3:
        labels = EDFFileReader.create_class_three(labels)
    elif class_num == 4:
        labels = EDFFileReader.create_class_four(labels)
    elif class_num == 5:
        labels = EDFFileReader.create_class_five(labels)

    return labels, signals


def main(class_num, in_data_type, batch_size, epochs, **args):
    # Printing settings for log
    print('Training with', class_num, 'number of classes and data type:', in_data_type)
    print('-------------------------')
    print('Hyper parameter settings:')
    print('Batch size:', batch_size)
    print('Epochs:', epochs, '\n\n')

    # Setting up the path for saving logs
    job_dir = os.getcwd()
    logs_path = job_dir + '/logs/tensorboard/'
    # class_num = 6

    # with tf.device('/device:GPU:0'):
    se = 42
    random.seed(se)

    # parse data
    labels, input_data = get_data(class_num, in_data_type)

    (trainX, testX, trainY, testY) = train_test_split(input_data, labels, test_size=0.3, random_state=se)
    (valX, testX, valY, testY) = train_test_split(testX, testY, test_size=0.5, random_state=se)

    # one hot encoding
    lb = LabelBinarizer()
    trainY = lb.fit_transform(trainY)
    testY = lb.transform(testY)
    valY = lb.transform(valY)

    # set training params (TODO maybe vary eventually)
    nb_classes = trainY.shape[1]
    # epochs = 30
    # batch_size = 128  # alt values: (32, 64, 128, 256)

    # set model_name
    model_name = 'model' + str(class_num) + in_data_type

    # create & train model
    NN = Model(nb_classes, se)

    # tensorboard = callbacks.TensorBoard(log_dir=logs_path + model_name, histogram_freq=10, write_graph=True,
    #                                    write_images=True)

    NN.m.fit(trainX, trainY, callbacks=[TrainValTensorBoard(write_graph=False)],
             batch_size=batch_size, epochs=epochs, shuffle=True, verbose=1, validation_data=(valX, valY))

    # evaluate model
    loss_acc = NN.m.evaluate(testX, testY, verbose=1)
    print("Test Loss is " + str(loss_acc[0]))
    print("Test Acc is " + str(loss_acc[1]))

    model_name += '.h5'
    NN.m.save(job_dir + '/models/' + model_name)


# App Runner
if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    #
    # # Input Args
    # parser.add_argument(
    #     '--job-dir',
    #     help='GCS location to write checkpoints and export models',
    #     required=True
    # )
    # parser.add_argument(
    #     '--class-num',
    #     help='Which label set is wanted',
    #     required=True
    # )
    #
    # args = parser.parse_args()
    # arguments = args.__dict__
    #
    # main(**arguments)
    main(int(sys.argv[1]), sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
