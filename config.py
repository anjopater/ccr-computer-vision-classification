
# config.py
import os

# Paths
#BASE_PATH = "/Users/antonio/Documents/projects/jupiterenv/datasets/LOBO_D_E_ORIGINAL_MIXED_argumented"
#C_PATH = os.path.join(BASE_PATH, "C")  # Control
#CCR_PATH = os.path.join(BASE_PATH, "CCR")  # Cancer


BASE_PATH = "/Users/antonio/Documents/projects/jupiterenv/datasets/FOTOS_HE_FIGADO_ORIGINAL_ORGANIZED"
C_PATH = os.path.join(BASE_PATH, "Controle")  # Control
CCR_PATH = os.path.join(BASE_PATH, "CR")  # Cancer

# PCA Components to Test
PCA_COMPONENTS = [30]

# Models to Test
MODELS = {
    # "Inceptionv3": "inceptionv3",
    #"ResNet50": "resnet50",
    "DenseNet121": "densenet121",
    #"EfficientNetB0": "efficientNetB0",
    #"ConvNeXtTiny": "convNeXtTiny"
}

IMAGES_SIZE_MODELS = {
   "inceptionv3" : 229,
   "densenet121": 224,
   "efficientNetB0": 224,
   "resnet50": 224,
   # "convNeXtTiny": 224

}

# GroupShuffleSplit Configuration
TEST_SIZE = 0.2  # 20% of the data for testing
RANDOM_STATE = 42

# Output File
RESULTS_FILE = "results.json"