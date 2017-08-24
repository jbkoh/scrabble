## Dependencies
Python 3.6>, 
From PIP: pycrfsuite, arrow..... (can't recall all of them for now.)

## File Descriptions
1. scrabble.py: Main file.
2. char2ir.py: CRF Mapping.
3. ir2tagsets.py: IR to TagSets learning and iteration functions.
4. hcc.py: Hierarchical Cluster Chain (It's named StructuredCC in the file currently.)

## Preprocessing (don't do it until you understand what they do.)
1. Generate ground truth (from schema\_map at Dropbox.)
2. Tokenize sentences with predefined rules (sentence\_normalizer.ipynb)
3. Label them with rules and experts (label\_with\_experts.ipynb)
4. Generate tokenized labels. (conv\_word\_labels\_to\_char\_labels.ipynb)
5. Generate ground truth file (ground\_truth\_gen.py <building>)

## How to use?
### Configuration options
 - -bl: Building list: list of source building names deliminated by comma (e.g., -bl ebu3b,bml).
 - -nl: Sample number list: list of sample numbers per building. The order should be same as bl. (e.g., -nl 200,1)
 - -t: Target building name: Name of the target building. (e.g., -t bml)
 - -c: Whether to use clustering for random selection or not (e.g., -c true)
 - -avg: How many times run experiments to get average? (e.g., -avg 5)
 - -iter: How many times to iterate the process? (e.g., -iter 10)
 - -d: Debug mode flag (e.g., -d false)
 - -ub: Whethre to use Brick when learning. (e.g., -ub true)
 - Note: Please refer scrabble.py to learn the other options.

### CRF only mode
CRF Test. No iteration. Character to phrases only (phrase == concatenated BILOU tags).
1. Learn a CRF model: ``` python scrabble.py learn\_crf -bl ebu3b,ap\_m -nl 200,10 -c true ```
2. Test a CRF model at a target building: python scrabble.py learn\_crf -bl ebu3b,ap\_m -nl 200,10 -c true


python scrabble.py 

