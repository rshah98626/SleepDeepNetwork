# SleepDeepNetwork
Deep Learning final project. Aims to replicate the study outlined in this paper: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6406978/

To run cloud_model.py, it takes two arguments:
1. class label settings (integer 2-6)
2. input data type (either 'fpz', 'eog', or 'both')

A sample command looks like this "python cloud_model.py 6 fpz"

To run the model in the cloud use nohup and pipe the output to another file.
Sample command "nohup python cloud_model.py 6 both > both6.txt &"
