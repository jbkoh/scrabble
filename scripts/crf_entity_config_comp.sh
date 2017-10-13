#!/bin/bash

cd ../
rm nohup/crfentity_conf_1.out nohup/crfentity_conf_2.out nohup/crfentity_conf_3.out 
nohup python scrabble.py crf_entity -bl ebu3b,ap_m -nl 200,5 -c true -neg true -ub true -t ap_m -iter 20 -post phraseutiltest -crfqs confidence -entqs phrase_util -nj 2 > nohup/crfentity_conf_1.out &
nohup python scrabble.py crf_entity -bl ebu3b,ap_m -nl 200,5 -c true -neg true -ub true -t ap_m -iter 20 -post tokenutiltest -crfqs confidence -entqs token_util -nj 2 > nohup/crfentity_conf_2.out &
nohup python scrabble.py crf_entity -bl ebu3b,ap_m -nl 200,5 -c true -neg true -ub true -t ap_m -iter 20 -post cmntest -crfqs confidence -entqs cmn -nj 2 > nohup/crfentity_conf_3.out &
