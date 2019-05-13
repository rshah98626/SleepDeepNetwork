import keras
from keras.models import Sequential
import keras.layers as KL
from keras import backend as K
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import random
from sklearn.preprocessing import LabelBinarizer

# set seed
se = 42
random.seed(se)

# read csv data in
file_path_data = 'firstData.csv'
file_path_label = 'firstData_Label.csv'
data_csv = pd.read_csv(file_path_data, delimiter=';')
labels_csv = pd.read_csv(file_path_label, delimiter=';')
# print(labels_csv.tail())
# print(data_csv.tail())

# set up label and data array
labels = np.array(labels_csv['Hypnogram'][:-1])  # TODO figure out why labels has one extra value
eog_in_data = np.array(data_csv['EOG horizontal[uV]'])
# eog_in_data = np.array([tuple(eog_in_data[i:i+3000]) for i in range(0, len(eog_in_data), 3000)])
eog_in_data = eog_in_data.reshape(-1, 3000, 1)
fpz_in_data = np.array(data_csv['EEG Fpz-Cz[uV]'])
fpz_in_data = fpz_in_data.reshape(-1, 3000, 1)
# fpz_in_data2 = np.array([tuple(fpz_in_data[i:i+3000]) for i in range(0, len(fpz_in_data), 3000)])
# print(np.array_equal(fpz_in_data1, fpz_in_data2))

# EOG test train split
(trainX, testX, trainY, testY) = train_test_split(eog_in_data, labels, test_size=0.3, random_state=se)
(valX, testX, valY, testY) = train_test_split(testX, testY, test_size=0.5, random_state=se)

# one hot encoding
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)
valY = lb.transform(valY)

# set training params
nb_classes = trainY.shape[1]
batch_size = 128  # TODO can change batch size to find best value (good vals are 32, 64, 128, 256)
epochs = 100
in_shape = (3000, 1)

print(trainX.shape)

# create model
eog_model = Sequential()
eog_model.add(KL.Conv1D(64, 5, strides=3, activation='relu', input_shape=in_shape))
eog_model.add(KL.Conv1D(128, 5, strides=1, activation='relu'))
eog_model.add(KL.MaxPool1D(pool_size=2, strides=2))
eog_model.add(KL.Dropout(0.2, seed=se))
eog_model.add(KL.Conv1D(128, 13, strides=1, activation='relu'))
eog_model.add(KL.Conv1D(256, 7, strides=1, activation='relu'))
eog_model.add(KL.MaxPool1D(pool_size=2, strides=2))
eog_model.add(KL.Conv1D(256, 7, strides=1, activation='relu'))
eog_model.add(KL.Conv1D(64, 4, strides=1, activation='relu'))
eog_model.add(KL.MaxPool1D(pool_size=2, strides=2))
eog_model.add(KL.Conv1D(32, 3, strides=1, activation='relu'))
eog_model.add(KL.Conv1D(64, 6, strides=1, activation='relu'))
eog_model.add(KL.MaxPool1D(pool_size=2, strides=2))
eog_model.add(KL.Conv1D(8, 5, strides=1, activation='relu'))
eog_model.add(KL.Conv1D(8, 2, strides=1, activation='relu'))
eog_model.add(KL.MaxPool1D(pool_size=2, strides=2))
eog_model.add(KL.Flatten())
eog_model.add(KL.Dense(64, activation='relu')) # TODO paper mentions drop = 0.2 so check if that means another dropout layer
eog_model.add(KL.Dense(nb_classes, activation='softmax'))

# print(eog_model.summary())

# choose optimizer
eog_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(lr=.0001, decay=.003),
              		metrics=['accuracy'])

# train model
history = eog_model.fit(trainX, trainY, batch_size=batch_size, epochs=epochs, shuffle=True, verbose=1,
						validation_data=(valX, valY))

# evaluate model
score = eog_model.evaluate(testX, testY, verbose=0)
print("Score is: ", score)
