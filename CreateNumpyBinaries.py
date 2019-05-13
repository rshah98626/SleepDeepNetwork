import EDFFileReader
import gc
import numpy as np
import sklearn.preprocessing as sk


def export_binary():
    export_edf()
    print("************\nMOVING TO EXPORT EDFX DATA\n************")
    export_edfx()


def export_edf():
    edf_signals, edf_labels = EDFFileReader.read_edf_data()
    np.save('np_files/edf_labels', edf_labels)
    gc.collect()

    fpz_edf = sk.normalize(edf_signals[:, 0].reshape(-1, 1), copy=False, axis=0).reshape(-1, 3000, 1)
    eog_edf = sk.normalize(edf_signals[:, 1].reshape(-1, 1), copy=False, axis=0).reshape(-1, 3000, 1)
    both_edf = np.add(edf_signals[:, 0], edf_signals[:, 0])
    np.save('np_files/edf_eog', eog_edf)
    np.save('np_files/edf_fpz', fpz_edf)
    gc.collect()

    both_edf = sk.normalize(both_edf.reshape(-1, 1), copy=False, axis=0).reshape(-1, 3000, 1)
    np.save('np_files/edf_both', both_edf)


def export_edfx():
    edfx_signals, edfx_labels = EDFFileReader.read_edfx_data()
    np.save('np_files/edfx_labels', edfx_labels)
    gc.collect()

    fpz_edfx = sk.normalize(edfx_signals[:, 0].reshape(-1, 1), copy=False, axis=0).reshape(-1, 3000, 1)
    eog_edfx = sk.normalize(edfx_signals[:, 1].reshape(-1, 1), copy=False, axis=0).reshape(-1, 3000, 1)
    both_edfx = np.add(edfx_signals[:, 0], edfx_signals[:, 0])
    np.save('np_files/edfx_eog', eog_edfx)
    np.save('np_files/edfx_fpz', fpz_edfx)
    gc.collect()

    both_edfx = sk.normalize(both_edfx.reshape(-1, 1), copy=False, axis=0).reshape(-1, 3000, 1)
    np.save('np_files/edfx_both', both_edfx)


if __name__ == "__main__":
    export_binary()
