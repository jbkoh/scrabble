Scrabble
========
Scrabble is a machine learning framework that can normalize unstructured metadata in Building Management Systems (BMS) such as point names into the structured metadata, [Brick](https://brickschema.org). It is evaluated in our own [paper](http://mesl.ucsd.edu/mesl-website/pubs/Jason_BuildSys2018Scrabble.pdf). This code base is implemented mainly for evaluating the results. If you would like to interactively add examples and get inferred results, please consider using [Plaster](https://github.com/plastering/plastering) together.

## General Procedure

It is best to run this with [Plaster](https://github.com/plastering/plastering). Here, we only describe the procedure to run Scrabble alone.

1. Store metadata and ground truth data following a MongoDB schema defined in ``scrabble/data_model.py``. Refer [Data Model](#datamodel) for detailed description.
2. Run ``scrabble`` with proper options.
3. Retrieve and process the results from the database. Example code: ``scripts/parse_scrabble_res.py``. (There will be better documents on this.)
4. Put labels for chosen examples. (This is only supported with Plaster.)

## Installation

### Dependencies
- Python 3
- MongoDB
- PIP packages: ``requirements.txt``
- Tested only on Linux. There can be issues with path names in Windows.

### Install
1. ``git clone https://github.com/jbkoh/scrabble.git``
2. ``cd scrabble``
3. ``python setup.py install`` (Note that it assumes to use TensorFlow instead of Theano for the backend of Keras.)

## File Descriptions (./)
1. `scripts/scrabble`: Main executable file. Once it's PIPped, you can run it in a command line.
2. `char2ir.py`: Learning mapping from characters to Brick Tags or Intermediate Representation (IR).
3. `ir2tagsets.py`: IR to TagSets learning and iteration functions.

<a name="datamodel"></a>
## Data Model
Data models in the form of mongoengine. Refer ``scrabble/data_model.py``. Each entity is associated with ``srcid``. A combination of ``srcid`` and ``building`` is unique in the database. Please refer ``TODO`` for ingesting a CSV file into MongoDB.

### RawMetadata
Raw metadata you can retrieve from existing systems like BMSes.
- Definition:
    ```python
    {
        "srcid": str # unique ID,
        "building": str # building name,
        "metadata": dict # key: metadata type, value: metadata
    }
    ```
- Example:
    ```json
    {
        "srcid": "1234",
        "building": "ebu3b",
        "metadata": {
            "VendorGivenName": "NAE2.VMA101.ZNT",
            "BACnetDescription": "Zone Temp",
            "BACnetUnit": 64
        }
    }
    ```
    ``VendorGivenName`` is commonly referred as point names also.

### LabeledMetadata
LabeledMetadata containing different types of labels per entity.
- Definition
    ```python
    {
        "srcid": str # unique ID,
        "building": str # building name,
        "fullparsing": dict # dict of list of tuples of character and its label.
        "tagsets": list # list of tagsets found in the metadata.
        "point_tagset": str # tagset for Point.
    }
    ```
- Example
    ```json
    {
        "srcid": "1234",
        "building": "ebu3b",
        "fullparsing": {
            "VendorGivenName": [
                ["N", "B_networkadapter-nae"],
                ["A", "I_networkadapter-nae"],
                ["E", "I_networkadapter-nae"],
                ["2", "B_leftidentifier"],
                [".", "O"],
                ["V", "B_vav"],
                ["M", "I_vav"],
                ["A", "I_vav"],
                [".", "O"],
                ["1", "B_leftidentifier"],
                ["0", "I_leftidentifier"],
                ["1", "I_leftidentifier"],
                [".", "O"],
                ["Z", "B_zone"],
                ["N", "I_zone"],
                ["T", "I_temperature"]
            ],
            "BACnetDescription": [
                ["Z", "B_Zone"],
                ["o", "I_Zone"],
                ["n", "I_Zone"],
                ["e", "I_Zone"],
                [" ", "O"],
                ["T", "B_temperature"],
                ["e", "I_temperature"],
                ["m", "I_temperature"],
                ["p", "I_temperature"]
            ]
        },
        "tagsets": ["networkadapter-nae", "vav", "zone_temperature_sensor"],
        "point_tagset": "zone_temperature_sensor",
    }
    ```
    In ``fullparsing``, you also need to put BIO tags to actual Brick Tags.
    B = Beginning
    I = Inside
    O = Outside
    If a character is in the starting position of a word (e.g., ``Z`` for ``ZN``), ``B_`` should be attached to actual label (e.g., ``B_Zone`` for ``Z``).
    Everything else with a Brick Tag should attach ``I_`` (``I_Zone`` for ``N`` in ``ZN``)
    If a character is not associated with any Brick Tag, label it as ``O``, which means nothing.


## How to Use it?

### Configuration options
 - -bl: Building list: list of source building names deliminated by comma (e.g., -bl ebu3b,bml).
 - -nl: Sample number list: list of sample numbers per building. The order should be same as bl. (e.g., -nl 200,1)
 - -t: Target building name: Name of the target building. (e.g., -t bml)
 - -c: Whether to use clustering for random selection or not (e.g., -c true)
 - -avg: How many times run experiments to get average? (e.g., -avg 5)
 - -iter: How many times to iterate the process? (e.g., -iter 10)
 - -d: Debug mode flag (e.g., -d false)
 - -ub: Whethre to use Brick when learning. (e.g., -ub true)
 - Note: Please refer ``scrabble --help`` to learn all the options.

### Best configuration
1. Active learning of two stages
    ```bash
    scrabble -task scrabble -bl ebu3b -nl 10 -ut true -neg true -c true -t ebu3b
    ```
    This learns the entire model of Scrabble. It will use only ``ebu3b`` to select more examples for the target ``ebu3b``, the same building. It initially select 10 examples randomly to initiate the model.

2. Active learning of two stages with a known building.
    ```bash
    scrabble -task scrabble -bl ap_m,ebu3b -nl 200,10 -ut true -neg true -c true -t ebu3b
    ```
    This does the same thing as above except using 200 examples of ``ap_m`` additionally.

3. Active learning of the first stage (CRF) only.
    ```bash
    scrabble -task char2ir -bl ap_m,ebu3b -nl 200,10 -c true -t ebu3b
    ```

4. Active learning of the second stage (MLP) only.
    ```bash
    scrabble -task ir2tagsets -bl ap_m,ebu3b -nl 200,10 -ut true -neg true -c true -t ebu3b
    ```


### References
1. *Scrabble: Transferrable Semi-Automated Semantic Metadata Normalization using Intermediate Representation*, BuildSys 2018.
2. *Plaster: An Integration, Benchmark, and Development Framework for Metadata Normalization Methods*, BuildSys 2018.
3. *Brick: Towards a unified metadata schema for buildings*, BuildSys 2016.
4. *Brick: Metadata schema for portable smart building applications*, AppliedEnergy 2018.

