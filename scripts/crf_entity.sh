#!/bin/bash

cd ../

rm nohup/crf_entity*.out

nohup python scrabble.py crf_entity -bl ebu3b,ap_m -nl 200,5 -c true -neg true -ub true -t ap_m -iter 20 -inc 10 -post 1 -nj 2 -crfqs confidence -entqs phrase_util > nohup/crf_entity1.out &
nohup python scrabble.py crf_entity -bl ap_m,bml -nl 200,5 -c true -neg true -ub true -t bml -iter 20 -inc 10 -post 1 -nj 2 -crfqs confidence -entqs phrase_util  > nohup/crf_entity2.out &
nohup python scrabble.py crf_entity -bl ebu3b,ghc -nl 200,5 -c true -neg true -ub true -t ghc -iter 20 -inc 10 -post 1 -nj 2  -crfqs confidence -entqs phrase_util  > nohup/crf_entity3.out &
nohup python scrabble.py crf_entity -bl ghc,ebu3b -nl 200,5 -c true -neg true -ub true -t ebu3b -iter 20 -inc 10 -post 1 -nj 2  -crfqs confidence -entqs phrase_util  > nohup/crf_entity4.out &
nohup python scrabble.py crf_entity -bl ebu3b,ap_m,bml -nl 200,200,5 -c true -neg true -ub true -t bml -iter 20 -inc 10 -post 1 -nj 2  -crfqs confidence -entqs phrase_util  > nohup/crf_entity5.out
