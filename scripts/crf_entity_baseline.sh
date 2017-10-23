#!/bin/bash

cd ../

nohup python naive_baseline.py -bl ebu3b,ap_m -nl 200,0 -t ap_m -iter 11 -inc 20 -nj 2 -avg 3 &
nohup python naive_baseline.py -bl ap_m,bml -nl 200,0 -t bml -iter 11 -inc 20 -nj 2 -avg 3 &
nohup python naive_baseline.py -bl ebu3b,ghc -nl 200,0 -t ghc -iter 11 -inc 20 -nj 2 -avg 3 &
nohup python naive_baseline.py -bl ghc,ebu3b -nl 200,0 -t ebu3b -iter 11 -inc 20 -nj 2 -avg 3 &
nohup python naive_baseline.py -bl ebu3b,bml,ap_m -nl 200,200,0 -t ap_m -iter 11 -inc 20 -avg 3 -nj 2 

#nohup python naive_baseline.py -bl ebu3b,ap_m -nl 200,1 -t ap_m -iter 10 -inc 20 -nj 2 -avg 3 &
#nohup python naive_baseline.py -bl ap_m,bml -nl 200,1 -t bml -iter 10 -inc 20 -nj 2 -avg 3 &
#nohup python naive_baseline.py -bl ebu3b,ghc -nl 200,5 -t ghc -iter 10 -inc 20 -nj 2 -avg 3 &
#nohup python naive_baseline.py -bl ghc,ebu3b -nl 200,5 -t ebu3b -iter 10 -inc 20 -nj 2 -avg 3 &
#nohup python naive_baseline.py -bl ebu3b,bml,ap_m -nl 200,200,5 -t ap_m -iter 10 -inc 20 -avg 3 -nj 2 

nohup python naive_baseline.py -bl ap_m -nl 1 -t ap_m -iter 11 -inc 20 -nj 2 -avg 3 > nohup/baseline6.out &
nohup python naive_baseline.py -bl bml -nl 1 -t bml -iter 11 -inc 20 -nj 2 -avg 3 > nohup/baseline7.out &
nohup python naive_baseline.py -bl ghc -nl 1 -t ghc -iter 11 -inc 20 -nj 2  -avg 3 > nohup/baseline8.out &
nohup python naive_baseline.py -bl ebu3b -nl 1 -t ebu3b -iter 11 -inc 20 -nj 2  -avg 3 > nohup/baseline9.out &
nohup python naive_baseline.py -bl ap_m -nl 1 -t ap_m -iter 11 -inc 20 -nj 2   -avg 3 > nohup/baseline10.out 
