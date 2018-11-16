#!/bin/bash

postfix=$1

#nohupfile="nohup.ir2tagset.ebu3b.ap_m.aug.ts.$postfix"
#(nohup python -u entry.py -task ir2tagsets -bl ap_m,ebu3b -nl 200,10 -t ebu3b -neg true -ub true -ut true -ts true -ct MLP -post $postfix > $nohupfile && slack_notify --msg "$nohupfile") &

#nohupfile="nohup.ir2tagset.ebu3b.ap_m.aug.$postfix"
#(nohup python -u entry.py -task ir2tagsets -bl ap_m,ebu3b -nl 200,10 -t ebu3b -neg true -ub true -ut true -ts false -ct MLP -post $postfix > $nohupfile && slack_notify --msg "$nohupfile") &
#
nohupfile="nohup.ir2tagset.ebu3b.aug.$postfix"
(nohup python -u entry.py -task ir2tagsets -bl ebu3b -nl 10 -t ebu3b -neg true -ub true -ut true -ct MLP -post $postfix > $nohupfile && slack_notify --msg "$nohupfile") &
#
#nohupfile="nohup.ir2tagset.ebu3b.noaug.$postfix"
#(nohup python -u entry.py -task ir2tagsets -bl ebu3b -nl 10 -t ebu3b -neg false -ub false -ut true -ct MLP -post $postfix > $nohupfile && slack_notify --msg "$nohupfile") &
#

#nohupfile="nohup.ir2tagset.ghc.ap_m.aug.$postfix"
#(nohup python -u entry.py -task ir2tagsets -bl ap_m,ghc -nl 200,10 -t ghc -neg true -ub true -ut true -ct MLP -post $postfix > $nohupfile && slack_notify --msg "$nohupfile") &

#nohupfile="nohup.ir2tagset.ghc.aug.$postfix"
#(nohup python -u entry.py -task ir2tagsets -bl ghc -nl 10 -t ghc -neg true -ub true -ut true -ct MLP -post $postfix > $nohupfile && slack_notify --msg "$nohupfile") &
#
#nohupfile="nohup.ir2tagset.ghc.noaug.$postfix"
#(nohup python -u entry.py -task ir2tagsets -bl ghc -nl 10 -t ghc -neg false -ub false -ut true -ct MLP -post $postfix > $nohupfile && slack_notify --msg "$nohupfile") &
