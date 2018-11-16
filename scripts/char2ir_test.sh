#!/bin/bash

algo="ap"
#algo="prev"
postfix="1"$algo
(nohup python -u entry.py -task char2ir -bl ebu3b,ghc -nl 400,10 -t ghc -crfalgo $algo -post $postfix > nohup.char2ir.ghc.ebu3b.400.$postfix && slack_notify --msg "char2ir ghc ebu3b 400 $postfix done at $(hostname)") &
(nohup python -u entry.py -task char2ir -bl ebu3b,ap_m -nl 400,10 -t ap_m -crfalgo $algo -post $postfix > nohup.char2ir.ap_m.ebu3b.400.$postfix && slack_notify --msg "char2ir ap_m ebu3b 400 $postfix done at $(hostname)") &
