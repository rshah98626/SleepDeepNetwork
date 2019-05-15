#!/bin/sh
conda activate myenv1
python cloud_model.py 2 fpz 128 50 > out_logs/edf_2_fpz.txt
python cloud_model.py 2 eog 128 50 > out_logs/edf_2_eog.txt
python cloud_model.py 2 both 128 50 > out_logs/edf_2_both.txt
python cloud_model.py 6 fpz 128 50 > out_logs/edf_6_fpz.txt
python cloud_model.py 6 eog 128 50 > out_logs/edf_6_eog.txt
python cloud_model.py 6 both 128 50 > out_logs/edf_6_both.txt
python cloud_model.py 5 fpz 128 50 > out_logs/edf_5_fpz.txt
python cloud_model.py 5 eog 128 50 > out_logs/edf_5_eog.txt
python cloud_model.py 5 both 128 50 > out_logs/edf_5_both.txt
python cloud_model.py 4 fpz 128 50 > out_logs/edf_4_fpz.txt
python cloud_model.py 4 eog 128 50 > out_logs/edf_4_eog.txt
python cloud_model.py 4 both 128 50 > out_logs/edf_4_both.txt
python cloud_model.py 3 fpz 128 50 > out_logs/edf_3_fpz.txt
python cloud_model.py 3 eog 128 50 > out_logs/edf_3_eog.txt
python cloud_model.py 3 both 128 50 > out_logs/edf_3_both.txt
