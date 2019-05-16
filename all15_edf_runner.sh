#!/bin/sh
echo Starting to cloud models
nohup python cloud_model.py 2 fpz 128 50 > out_logs/edf_2_fpz.txt &
wait
echo Finished 2 fpz 
nohup python cloud_model.py 2 eog 128 50 > out_logs/edf_2_eog.txt &
wait
echo Finished 2 eog
nohup python cloud_model.py 2 both 128 50 > out_logs/edf_2_both.txt &
wait
echo Finished 2 both
nohup python cloud_model.py 6 fpz 128 50 > out_logs/edf_6_fpz.txt &
wait
echo Finished 6 fpz
nohup python cloud_model.py 6 eog 128 50 > out_logs/edf_6_eog.txt &
wait
echo Finished 6 eog
nohup python cloud_model.py 6 both 128 50 > out_logs/edf_6_both.txt &
wait
echo Finished 6 both
nohup python cloud_model.py 5 fpz 128 50 > out_logs/edf_5_fpz.txt &
wait
echo Finished 5 fpz
nohup python cloud_model.py 5 eog 128 50 > out_logs/edf_5_eog.txt &
wait
echo Finished 5 eog
nohup python cloud_model.py 5 both 128 50 > out_logs/edf_5_both.txt &
wait
echo Finished 5 both
nohup python cloud_model.py 4 fpz 128 50 > out_logs/edf_4_fpz.txt &
wait
echo Finished 4 fpz
nohup python cloud_model.py 4 eog 128 50 > out_logs/edf_4_eog.txt &
wait
echo Finished 4 eog
nohup python cloud_model.py 4 both 128 50 > out_logs/edf_4_both.txt &
wait
echo Finished 4 both
nohup python cloud_model.py 3 fpz 128 50 > out_logs/edf_3_fpz.txt &
wait
echo Finished 3 fpz
nohup python cloud_model.py 3 eog 128 50 > out_logs/edf_3_eog.txt &
wait
echo Finished 3 eog
nohup python cloud_model.py 3 both 128 50 > out_logs/edf_3_both.txt &
wait
echo Finished 3 both
