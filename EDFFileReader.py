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
    labels = g.readSignal(0).astype(int)  # Hypnogram data

    # Prove, if signals or labels is longer (unlabeled data vs labels for
    # data that does not exist) and cut data accordingly.
    signals, labels = cut(signals, labels)

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

    # Prove, if signals or labels is longer (unlabeled data vs labels for
    # data that does not exist) and cut data accordingly.
    signals, labels = cut(signals, labels)

    return signals, labels


def cut(signals, labels):
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
    files = get_edf_files()

    # Iterate over edf files:
    print("\nReading edf files:")
    print("\nFiles", files[1], "and", files[0], "...")
    signals, labels = read_edf(files[1], files[0])

    signals, labels = cleanup(signals, labels)

    for i in range(2, len(files), 2):
        print("\nFiles", files[i + 1], " and ", files[i], "...")
        s, l = read_edf(files[i + 1], files[i])

        s, l = cleanup(s, l)
        signals, labels = concat(signals, labels, s, l)

    # Read all edfx files:
    files = get_edfx_files()

    # Iterate over edfx files:
    print("\nReading edfx files:")
    for i in range(0, len(files), 2):
        print("\nFiles ", files[i], " and ", files[i + 1],"...")
        s, l = read_edfx(files[i], files[i + 1])

        s, l = cleanup(s, l)
        signals, labels = concat(signals, labels, s, l)

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


def get_edf_files():
    path = 'edf-files/edf/'
    files = sorted(os.listdir(path))
    for i in range(0, len(files)):
        files[i] = path + files[i]
    return files


def get_edfx_files():
    path = 'edf-files/edfx/'
    files = sorted(os.listdir(path))

    # Remove the following files as the Hypnograms don't match the signal recordings
    # (Tested with  prove_starttime()
    files_faulty = ['ST7021J0-PSG.edf', 'ST7021JM-Hypnogram.edf',
                    'ST7071J0-PSG.edf', 'ST7071JA-Hypnogram.edf',
                    'ST7092J0-PSG.edf', 'ST7092JE-Hypnogram.edf',
                    'ST7131J0-PSG.edf', 'ST7131JR-Hypnogram.edf',
                    'ST7132J0-PSG.edf', 'ST7132JR-Hypnogram.edf',
                    'ST7141J0-PSG.edf', 'ST7141JE-Hypnogram.edf',
                    'ST7142J0-PSG.edf', 'ST7142JE-Hypnogram.edf']
    files = sorted(list(set(files) - set(files_faulty)))
    for i in range(0, len(files)):
        files[i] = path + files[i]
    return files


def prove_starttime():
    # Read all edf files:
    files = get_edf_files()

    for i in range(0, len(files), 2):
        print('Testing', files[i + 1], 'and', files[i], '...')
        signals = pyedflib.EdfReader(files[i + 1])
        hypnogram = pyedflib.EdfReader(files[i])

        signals_start = signals.getStartdatetime()
        hypnogram_start = hypnogram.getStartdatetime()
        if signals_start == hypnogram_start:
            print('OK: signals_start == hypnogram_start')
        else:
            print('NOT OK: Signal start time:', signals_start,
                  ', Hypnogram start time:', hypnogram_start)

    # Read all edfx files:
    files = get_edfx_files()

    for i in range(0, len(files), 2):
        print('Testing', files[i], 'and', files[i + 1], '...')
        try:
            signals = pyedflib.EdfReader(files[i])
        except OSError:
            print("Error occurred when opening", files[i], ":", sys.exc_info()[1], '\n')
        try:
            hypnogram = pyedflib.EdfReader(files[i + 1])
        except OSError:
            print("Error occurred when opening", files[i + 1], ":", sys.exc_info()[1], '\n')

        signals_start = signals.getStartdatetime()
        hypnogram_start = hypnogram.getStartdatetime()
        if signals_start == hypnogram_start:
            print('OK: signals_start == hypnogram_start')
        else:
            print('NOT OK: Signal start time:', signals_start,
                  'Hypnogram start time:', hypnogram_start)
            print('Signal recording duration:', signals.file_duration,
                  ', Hypnogram duration', hypnogram.file_duration)
            if len(hypnogram.readAnnotations()[0].astype(int)) > 0:
                print('Duration data:', hypnogram.readAnnotations()[0].astype(int)[-1], '\n')
            else:
                print('Duration data: 0\n')


def create_class_six(labels):
    return labels


def create_class_five(labels):
    # SWS is #6
    for i in range(len(labels)):
        if labels[i] == 3 or labels[i] == 4:
            labels[i] = 6
    return labels


def create_class_four(labels):
    # light sleep is #7
    for i in range(len(labels)):
        if labels[i] == 3 or labels[i] == 4:
            labels[i] = 6
        if labels[i] == 1 or labels[i] == 2:
            labels[i] = 7
    return labels


def create_class_three(labels):
    # NREM is #8
    for i in range(len(labels)):
        if labels[i] == 1 or labels[i] == 2 or labels[i] == 3 or labels[i] == 4:
            labels[i] = 8
    return labels


def create_class_two(labels):
    # Sleep is #9
    for i in range(len(labels)):
        if labels[i] == 1 or labels[i] == 2 or labels[i] == 3 or labels[i] == 4 or labels[i] == 5:
            labels[i] = 9
    return labels


# all_signals, all_labels = read_all()
# classTwo = createClassTwo(all_labels)
# create_class_five(all_labels)
