# SleepDeepNetwork
Deep Learning final project. Aims to replicate the study outlined in this paper: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6406978/

To run cloud_model.py, it takes four arguments:
1. dataset (either 'edf' or 'edfx')
2. class label settings (integer 2-6)
3. input data type (either 'fpz', 'eog', or 'both')
4. batch size (typically 128, alt values: 32, 64, 128, 256)
5. number of epochs
6. name of the tensorboard directory (not needed, default: 'model' + class label setting + input data type, e.g. model6fpz)

A sample command looks like this "python cloud_model.py 6 fpz 128 100"

To run the model in the cloud use nohup and pipe the output to another file.
Sample command "nohup python cloud_model.py 6 both 128 30 > both6.txt &"
