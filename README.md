## Dependencies
Python 3.6>, MongoDB
From PIP: pycrfsuite, arrow..... (can't recall all of them for now.)

## File Descriptions (./)
1. scrabble.py: Main file.
2. char2ir.py: CRF Mapping.
3. ir2tagsets.py: IR to TagSets learning and iteration functions.
4. hcc.py: Hierarchical Cluster Chain (It's named StructuredCC in the file currently.)

## Metadata Descriptions (./metadata/)
Metadata files are generated with manual preprocessing stage. Please stick to existing ones. Words are characters with unit meaning. A string is a word if the meaning of the token is not interpretable for the users when the token is separated more
1. xxx\_char\_sentence\_dict.json: raw sentences in char list.
2. xxx\_sentence\_dict.json: raw sentences in tokens (~=words).
4. xxx\_label\_dict.json: Tags label per token. This should be aligned with 2.
3. xxx\_ground\_truth.json: Ground truth labels of tokens (~=words).
4. xxx\_true\_tagsets.json: Ground truth TagSets.

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

### Preprocessing (don't do it until you understand what they do.)
1. Generate ground truth (from schema\_map at Dropbox.)
2. Tokenize sentences with predefined rules (sentence\_normalizer.ipynb)
3. Label them with rules and experts (label\_with\_experts.ipynb)
4. Generate tokenized labels. (conv\_word\_labels\_to\_char\_labels.ipynb)
5. Generate ground truth file (ground\_truth\_gen.py <building>)


### Char2IR (CRF) only mode
CRF Test. No iteration. Character to phrases only (phrase == concatenated BILOU tags).
1. Learn a CRF model: ```python scrabble.py learn\_crf -bl ebu3b,bml -nl 200,10 -c true ```
2. Test a CRF model at a target building: ```python scrabble.py predict\_crf -bl ebu3b,bml -nl 200,10 -c true -t bml```

Note that CRF models and results are stored in MongoDB. Some of them are cached in ``` ./results ``` folder.


### IR2TagSet only mode
IR2 TagSet test. Iteration is possible.
1. Learn and test a model: ```python scrabble.py entity -bl ebu3b,bml -nl 200,10 -c true -t bml ```
2. Iteration: ```python scrabble.py entity -bl ebu3b,bml -nl 200,10 -c true -t bml  -iter 10```
2. Average : ```python scrabble.py entity -bl ebu3b,bml -nl 200,10 -c true -t bml  -avg 10```

### Iterating altogether
Char &rarr; IR &rarr; TagSet, and then iteration.
1. Learn and test: ```python scrabble.py crf_entity -bl ebu3b,bml -nl 200,10 -c true -t bml ```
