import EDFFileReader
import gc
import numpy as np
import sys
import sklearn.preprocessing as sk


def export_binary(normalize_per_patient):
    export_edf(normalize_per_patient)
    print("************\nMOVING TO EXPORT EDFX DATA\n************")
    export_edfx(normalize_per_patient)


def export_edf(normalize_per_patient):
    edf_signals, edf_labels = EDFFileReader.read_edf_data(normalize_per_patient)
    np.save('np_files/edf_labels', edf_labels)
    gc.collect()

    # TODO figure out if we do normalization for all signal data with sk.normalize or per patient
    # fpz_edf = sk.normalize(edf_signals[:, 0].reshape(-1, 1), copy=False, axis=0).reshape(-1, 3000, 1)
    # eog_edf = sk.normalize(edf_signals[:, 1].reshape(-1, 1), copy=False, axis=0).reshape(-1, 3000, 1)
    # both_edf = sk.normalize(both_edf.reshape(-1, 1), copy=False, axis=0).reshape(-1, 3000, 1)

    if not normalize_per_patient:
        edf_signals = sk.scale(edf_signals)

    fpz_edf = edf_signals[:, 0].reshape(-1, 3000, 1)
    eog_edf = edf_signals[:, 1].reshape(-1, 3000, 1)
    both_edf = edf_signals[:, 2].reshape(-1, 3000, 1)

    if normalize_per_patient:
        np.save('np_files/edf_eog', eog_edf)
        np.save('np_files/edf_fpz', fpz_edf)
        np.save('np_files/edf_both', both_edf)
    else:
        np.save('np_files/edf_eog_norm', eog_edf)
        np.save('np_files/edf_fpz_norm', fpz_edf)
        np.save('np_files/edf_both_norm', both_edf)
    gc.collect()


def export_edfx(normalize_per_patient):
    edfx_signals, edfx_labels = EDFFileReader.read_edfx_data(normalize_per_patient)
    np.save('np_files/edfx_labels', edfx_labels)
    gc.collect()

    # TODO figure out if we do normalization for all signal data with sk.normalize or per patient
    # fpz_edfx = sk.normalize(edfx_signals[:, 0].reshape(-1, 1), copy=False, axis=0).reshape(-1, 3000, 1)
    # eog_edfx = sk.normalize(edfx_signals[:, 1].reshape(-1, 1), copy=False, axis=0).reshape(-1, 3000, 1)
    # both_edfx = np.add(edfx_signals[:, 0], edfx_signals[:, 0])
    # both_edfx = sk.normalize(both_edfx.reshape(-1, 1), copy=False, axis=0).reshape(-1, 3000, 1)

    if not normalize_per_patient:
        edfx_signals = sk.scale(edfx_signals, copy=False, axis=0)

    fpz_edfx = edfx_signals[:, 0].reshape(-1, 3000, 1)
    eog_edfx = edfx_signals[:, 1].reshape(-1, 3000, 1)
    both_edfx = edfx_signals[:, 2].reshape(-1, 3000, 1)

    if normalize_per_patient:
        np.save('np_files/edfx_eog', eog_edfx)
        np.save('np_files/edfx_fpz', fpz_edfx)
        np.save('np_files/edfx_both', both_edfx)
    else:
        np.save('np_files/edfx_eog_norm', eog_edfx)
        np.save('np_files/edfx_fpz_norm', fpz_edfx)
        np.save('np_files/edfx_both_norm', both_edfx)
    gc.collect()


if sys.argv[1].lower() == 'true':
    export_binary(True)
elif sys.argv[1].lower() == 'false':
    export_binary(False)
else:
    raise Exception(sys.argv[1] + ' is an invalid input. Should be either true or false.')
