#!/bin/bash

# Create the  directory


# Create subdirectories for models, utils, tests
mkdir -p models
mkdir -p utils
mkdir -p tests

# Create the __init__.py files in each directory
touch models/__init__.py
touch utils/__init__.py

# Create model files in the models directory
touch models/resnet18.py
touch models/resnet34.py
touch models/resnet50.py

# Create utility function files in the utils directory
touch utils/data_loader.py
touch utils/feature_extractor.py
touch utils/evaluator.py
touch utils/logger.py

# Create test files for models in the tests directory
touch tests/test_resnet18.py
touch tests/test_resnet34.py
touch tests/test_resnet50.py

# Create the main.py script and configuration file
touch main.py
touch config.py

# Create the results.json file
touch results.json
