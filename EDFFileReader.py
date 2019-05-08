import numpy as np
import os
import pyedflib  # https://pyedflib.readthedocs.io/en/latest/#requirements
import sys


def read_edf(signal_path, hypnogram_path):
    f = pyedflib.EdfReader(signal_path)
    # print(f.getSignalLabels())
    signals = np.zeros((f.getNSamples()[0], 2))
    signals[:, 0] = f.readSignal(0)  # EEG Fpz-Cz
    signals[:, 1] = f.readSignal(2)  # EOG horizontal

    g = pyedflib.EdfReader(hypnogram_path)
    # print(g.getSignalLabels())
    labels = g.readSignal(0).astype(int) #  Hypnogram data

    seconds_signals = signals.shape[0] // 100
    seconds_lables = labels.shape[0] * 30
    if seconds_signals != seconds_lables:
        if seconds_signals > seconds_lables:  # Cut off signals
            signals = signals[0: seconds_lables * 100, :]
            print(seconds_signals - seconds_lables, "seconds have been cut from signals...")
        else:  # Cuf off labels
            labels = labels[0: seconds_signals // 30]
            signals = signals[0: len(labels) * 3000, :]
            print(seconds_lables - seconds_signals, "seconds have been cut from labels...")

    return signals, labels


def read_edfx(signal_path, hypnogram_path):
    f = pyedflib.EdfReader(signal_path)
    # print(f.getSignalLabels())
    signals = np.zeros((f.getNSamples()[0], 2))
    signals[:, 0] = f.readSignal(0)  # EEG Fpz-Cz
    signals[:, 1] = f.readSignal(2)  # EOG horizontal
    signals = signals

    g = pyedflib.EdfReader(hypnogram_path)
    hypno_data = g.readAnnotations()
    hypno_intervals = hypno_data[0].astype(int)
    hypno_labels = hypno_data[2]

    labels = np.zeros(hypno_intervals[-1] // 30)
    iterator = 0
    for second in range(0, hypno_intervals[-1], 30):
        if second >= hypno_intervals[iterator + 1]:
            iterator += 1

        label = label_switcher(hypno_labels[iterator])
        if label == 'Invalid label':
            print("Error: Unknown label: " + hypno_labels[iterator])
            sys.exit(1)
        else:
            labels[second // 30] = label

    seconds_signals = signals.shape[0] // 100
    seconds_lables = labels.shape[0] * 30
    if seconds_signals != seconds_lables:
        if seconds_signals > seconds_lables:  # Cut off signals
            signals = signals[0: seconds_lables * 100, :]
            print(seconds_signals - seconds_lables, "seconds have been cut from signals...")
        else:  # Cuf off labels
            labels = labels[0: seconds_signals // 30]
            print(seconds_lables - seconds_signals, "seconds have been cut from labels...")

    return signals, labels

def label_switcher(label):
    # Transform the labeling of the EDFX database to the same labeling as it was used in
    # the EDF database. From the documentation of the EDF database:
    # "The sleep stages W, 1, 2, 3, 4, R, M and 'unscored' are coded in the file as binaries
    # 0, 1, 2, 3, 4, 5, 6 and 9 according to Q16.3 of the EDF-FAQ"
    return {
        'Sleep stage W': 0,
        'Sleep stage 1': 1,
        'Sleep stage 2': 2,
        'Sleep stage 3': 3,
        'Sleep stage 4': 4,
        'Sleep stage R': 5,
        'Movement time': 6,
        'Sleep stage ?': 9
    }.get(label, 'Invalid label')


def read_all():
    # Read all edf files:
    path = '../edf-files/edf/'
    files = os.listdir(path)
    for i in range(0, len(files)):
        files[i] = path + files[i]
    # print(files)

    # Iterate over edf files:
    print("\nReading in edf files:")
    print("\nFiles", files[1], "and", files[0], "...")
    signals, labels = read_edf(files[1], files[0])

    signals, labels = cleanup(signals, labels)

    for i in range(2, len(files), 2):
        print("\nFiles", files[i + 1], " and ", files[i], "...")
        s, l = read_edf(files[i + 1], files[i])

        s, l = cleanup(s, l)
        signals, labels = concat(signals, labels, s, l)

    # TODO fix reading of edfx files
    # # Read all edfx files:
    # path = '../edf-files/edfx/'
    # files = os.listdir(path)
    # for i in range(0, len(files)):
    #     files[i] = path + files[i]
    #
    # # Iterate over edfx files:
    # print("\nReading in edfx files:")
    # for i in range(0, len(files), 2):
    #     print("\nFiles ", files[i], " and ", files[i + 1],"...")
    #     s, l = read_edfx(files[i], files[i + 1])
    #
    #     s, l = cleanup(s, l)
    #     signals, labels = concat(signals, labels, s, l)

    return signals, labels


def cleanup(signals, labels):
    # Remove labels 6 and 9 (Movement and unlabeled)
    s = signals.copy()
    l = labels.copy()
    already_deleted = 0
    for i in range(0, len(labels)):
        if labels[i] == 6 or labels[i] == 9:
            l = np.delete(l, i - already_deleted)
            s = np.delete(s, range((i - already_deleted) * 3000, (i - already_deleted + 1) * 3000), axis=0)
            already_deleted += 1

    if already_deleted != 0:
        print(already_deleted, "label(s) deleted.")

    if np.shape(l)[0] * 3000 != np.shape(s)[0]:
        print("Something went wrong:")
        print("Shape signals:", np.shape(s))
        print("Shape labels:", np.shape(l))

    # Normalize signals
    mean = np.mean(s, axis=0)
    print("Calculated mean:", mean)
    std = np.std(s, axis=0)
    print("Calculated std:", std)
    s = (s - mean) / std

    return s, l


def concat(signals, labels, s, l):
    # Concatenate
    return np.concatenate((signals, s), axis=0), \
           np.concatenate((labels, l), axis=0)


# all_signals, all_labels = read_all()
