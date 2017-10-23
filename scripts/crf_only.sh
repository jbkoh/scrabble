#!/bin/bash

cd ../

rm nohup/crf_only*.out

nohup python scrabble.py iter_crf -bl ebu3b,ap_m -c true -inc 10 -iter 20 -nl 1000,5 -t ap_m -post 0 > nohup/crf_only7.out &
nohup python scrabble.py iter_crf -bl ebu3b,ap_m -c true -inc 10 -iter 20 -nl 1000,5 -t ap_m -post 0 > nohup/crf_only8.out &
nohup python scrabble.py iter_crf -bl ebu3b,ap_m -c true -inc 10 -iter 20 -nl 1000,5 -t ap_m -post 0 > nohup/crf_only9.out &
nohup python scrabble.py iter_crf -bl ebu3b,ap_m -c true -inc 10 -iter 20 -nl 200,5 -t ap_m -post 0 > nohup/crf_only1.out &
nohup python scrabble.py iter_crf -bl ebu3b,ap_m -c true -inc 10 -iter 20 -nl 200,5 -t ap_m -post 0 > nohup/crf_only2.out &
nohup python scrabble.py iter_crf -bl ebu3b,ap_m -c true -inc 10 -iter 20 -nl 200,5 -t ap_m -post 0 > nohup/crf_only3.out &
nohup python scrabble.py iter_crf -bl ap_m -c true -inc 10 -iter 20 -nl 5 -t ap_m -post 0 > nohup/crf_only4.out &
nohup python scrabble.py iter_crf -bl ap_m -c true -inc 10 -iter 20 -nl 5 -t ap_m -post 0 > nohup/crf_only5.out &
nohup python scrabble.py iter_crf -bl ap_m -c true -inc 10 -iter 20 -nl 5 -t ap_m -post 0 > nohup/crf_only6.out &
#nohup python scrabble.py iter_crf -bl ap_m -c true -inc 10 -iter 20 -nl 5 -t ap_m -post 3
