import EDFFileReader
import gc
import numpy as np


def export_binary():
    export_edf()
    print("************\nMOVING TO EXPORT EDFX DATA\n************")
    export_edfx()


def export_edf():
    edf_signals, edf_labels = EDFFileReader.read_edf_data()
    np.save('edf_labels', edf_labels)
    gc.collect()

    fpz_edf = edf_signals[:, 0].reshape(-1, 3000, 1)
    eog_edf = edf_signals[:, 1].reshape(-1, 3000, 1)
    gc.collect()

    both_edf = np.add(fpz_edf, eog_edf)
    both_edf = (both_edf - np.mean(both_edf, axis=0)) / np.std(both_edf, axis=0)
    both_edf = both_edf.reshape(-1, 3000, 1)

    np.save('edf_eog', eog_edf)
    np.save('edf_fpz', fpz_edf)
    np.save('edf_both', both_edf)


def export_edfx():
    edfx_signals, edfx_labels = EDFFileReader.read_edfx_data()
    np.save('edfx_labels', edfx_labels)
    gc.collect()

    fpz_edfx = edfx_signals[:, 0].reshape(-1, 3000, 1)
    eog_edfx = edfx_signals[:, 1].reshape(-1, 3000, 1)
    gc.collect()

    both_edfx = np.add(fpz_edfx, eog_edfx)
    both_edfx = (both_edfx - np.mean(both_edfx, axis=0)) / np.std(both_edfx, axis=0)
    both_edfx = both_edfx.reshape(-1, 3000, 1)

    np.save('edfx_eog', eog_edfx)
    np.save('edfx_fpz', fpz_edfx)
    np.save('edfx_both', both_edfx)


if __name__ == "__main__":
    export_binary()
