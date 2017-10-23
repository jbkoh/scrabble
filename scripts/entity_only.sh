#!/bin/bash


cd ../

rm nohup/entity_only*.out

nohup python scrabble.py entity -bl ap_m -nl 15 -c true -neg false -ub false -t ap_m -entqs phrase_util -post nosource_nosa1 -iter 21 -inc 10 -nj 4 > nohup/entity_only1.out &
nohup python scrabble.py entity -bl ap_m -nl 15 -c true -neg true -ub true -t ap_m -entqs phrase_util -post nosource_sa1 -iter 21 -inc 10 -nj 4 > nohup/entity_only2.out & 
nohup python scrabble.py entity -bl ebu3b,ap_m -nl 200,0 -c true -neg true -ub true -t ap_m -crfqs confidence -entqs phrase_util -post source_sa -iter 21 -inc 10 -nj 4 > nohup/entity_only3.out 


nohup python scrabble.py entity -bl ebu3b -nl 15 -c true -neg false -ub false -t ebu3b -entqs phrase_util -post nosource_nosa1 -iter 21 -inc 10 -nj 4 > nohup/entity_only4.out &
nohup python scrabble.py entity -bl ebu3b -nl 15 -c true -neg true -ub true -t ebu3b -entqs phrase_util -post nosource_sa1 -iter 21 -inc 10 -nj 4 > nohup/entity_only5.out & 
nohup python scrabble.py entity -bl ghc,ebu3b -nl 200,0 -c true -neg true -ub true -t ebu3b -crfqs confidence -entqs phrase_util -post source_sa -iter 21 -inc 10 -nj 4 > nohup/entity_only6.out &
