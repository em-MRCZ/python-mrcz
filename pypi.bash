#!/bin/bash

# SDIST
source activate py34
python setup.py register -r pypi
python setup.py sdist upload -r pypi
