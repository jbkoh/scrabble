#!/bin/bash

cd ../
rm nohup/crf_entity_nosource*.out

nohup python scrabble.py crf_entity -bl ap_m -nl 5 -c true -neg true -ub true -t ap_m -iter 11 -inc 20 -post nosource1 -nj 2 -crfqs confidence -entqs phrase_util > nohup/crf_entity_nosource1.out &
nohup python scrabble.py crf_entity -bl bml -nl 5 -c true -neg true -ub true -t bml -iter 11 -inc 20 -post nosource1 -nj 2 -crfqs confidence -entqs phrase_util  > nohup/crf_entity_nosource2.out &
nohup python scrabble.py crf_entity -bl ghc -nl 5 -c true -neg true -ub true -t ghc -iter 11 -inc 20 -post nosource1 -nj 2  -crfqs confidence -entqs phrase_util  > nohup/crf_entity_nosource3.out &
nohup python scrabble.py crf_entity -bl ebu3b -nl 5 -c true -neg true -ub true -t ebu3b -iter 11 -inc 20 -post nosource1 -nj 2  -crfqs confidence -entqs phrase_util  > nohup/crf_entity_nosource4.out &
