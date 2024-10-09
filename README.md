# traditional-feature-extraction

contact: slwanna@utexas.edu

TODO

## Setup Instructions
- ```$ conda env create -f environment.yml```
- ```$ conda activate trad-feature-extraction```
- ```$ make dev_install```
- ```$ python -m spacy download en_core_web_sm```
- ```$ python -c "import nltk; nltk.download('punkt_tab')"```

## How to Run
This package assumes you also have the ```dataset-download-scripts``` package. This is because line 33 in this package's ```__main__.py``` references the ```metadata.json``` file in the ```dataset-download-scripts``` package.

- Alter or setup your preferred experiment configuration in ```traditional-feature-extraction/config.ini```.
- ```$ python -m traditional_feature_extraction```

**WARNING: THIS SCRIPT ASSUMES ACCESS TO GPU RESOURCES. ADJUST BATCH SIZES IN SOME OF THE FEATURE ACQUISITION FUNCTIONS TO BETTER SUIT YOUR OWN MACHINES.**

## Before you commit!

1. Ensure you code "works" otherwise save to an appropriate branch
2. run ```$ make format``` 
3. run ```$ make lint```   
