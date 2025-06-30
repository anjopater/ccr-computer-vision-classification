
# config.py
import os

BASE_PATH = "/Users/antonio/Documents/projects/jupiterenv/datasets/HE_DATASET_LIVER"
C_PATH = os.path.join(BASE_PATH, "Controle")  # Control
CCR_PATH = os.path.join(BASE_PATH, "CRC")  # Cancer

# PCA Components to Test
PCA_COMPONENTS = [80, 60]

# Models to Test
MODELS = {
   #"Inceptionv3": "inceptionv3",
   #"ResNet50": "resnet50",
   #"DenseNet121": "densenet121",
   #"EfficientNetB0": "efficientNetB0",
   #"ConvNeXtTiny": "convNeXtTiny"
   "handcrafted": "handcrafted"   # ← new key
}

IMAGES_SIZE_MODELS = {
   "inceptionv3" : 229,
   "densenet121": 224,
   "efficientNetB0": 224,
   "resnet50": 224,
}
# Output File
RESULTS_FILE = "results.json"
