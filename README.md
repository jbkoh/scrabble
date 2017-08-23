
# Ground truth generation instruction
1. Generate ground truth (from schema\_map at Dropbox.)
2. Tokenize sentences with predefined rules (sentence\_normalizer.ipynb)
3. Label them with rules and experts (label\_with\_experts.ipynb)
4. Generate tokenized labels. (conv\_word\_labels\_to\_char\_labels.ipynb)
5. Generate ground truth file (ground_truth_gen.py <building>)
6. Learn CRF model (python scrabble_hierarchy.py learn --bl=bld1,bld2, -nl=200,10) #Select 200 samples from bld1 and 10 from bld2 to construct a CRF model.)
7. Learn Brick model.
8. Test.
