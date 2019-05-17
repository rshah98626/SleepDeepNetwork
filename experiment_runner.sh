#!/bin/sh
echo Starting to cloud models
nohup python cloud_model.py edfx 6 fpz 128 100 patient 1 0 experiment5 > out_logs/experiment5.txt &
wait
echo Finished Experiment 5
nohup python cloud_model.py edfx 6 eog 128 30 patient 1 0 experiment6 > out_logs/experiment6.txt &
wait
echo Finished Experiment 6
nohup python cloud_model.py edfx 6 both 128 30 patient 1 0 experiment7 > out_logs/experiment7.txt &
wait
echo Finished Experiment 7
# nohup python cloud_model.py edfx 6 fpz 128 30 patient 1 experiment8 > out_logs/experiment8.txt &
# wait
# echo Finished Experiment 8
# nohup python cloud_model.py edf 6 fpz 64 50 patient 0 experiment9 > out_logs/experiment9.txt &
# wait
# echo Finished Experiment 9
# nohup python cloud_model.py edf 6 fpz 256 50 patient 0 experiment10 > out_logs/experiment10.txt &
# wait
# echo Finished Experiment 10
# nohup python cloud_model.py edf 6 fpz 128 50 patient 1 1 experiment11 > out_logs/experiment11.txt
# wait
# echo Finished Experiment 11
nohup python cloud_model.py edfx 6 fpz 128 100 patient 1 1 experiment12 > out_logs/experiment12.txt &
wait
echo Finished Experiment 12
nohup python cloud_model.py edf 6 fpz 512 50 patient 0 0 experiment13 > out_logs/experiment13.txt &
wait
echo Finished Experiment 13
nohup python cloud_model.py edf 6 fpz 1024 50 patient 0 0 experiment14 > out_logs/experiment14.txt &
wait
echo Finished Experiment 14