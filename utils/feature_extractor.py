# utils/feature_extractor.py
import numpy as np
from PIL import Image
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications import DenseNet121, EfficientNetB0, ConvNeXtTiny
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.inception_v3 import preprocess_input as inception_preprocess
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from tensorflow.keras import layers, models
import warnings
from sklearn.exceptions import ConvergenceWarning

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from config import IMAGES_SIZE_MODELS


def getPreprocess_input(model_name):
    if model_name == "resnet50":
        return resnet_preprocess
    elif model_name == "efficientNetB0":
        return efficientnet_preprocess
    elif model_name == "densenet121":
        return densenet_preprocess
    elif model_name == "inceptionv3":
        return inception_preprocess

    
def extract_cnn_features(image_paths, model_name):
    print("EXTRATING FEATURES WITH MODEL")
    print(model_name == "resnet50")

    if model_name == "efficientNetB0":
        base_model = EfficientNetB0(weights='imagenet', include_top=False, pooling='avg')
    elif model_name == "densenet121":
        base_model = DenseNet121(weights='imagenet', include_top=False, pooling='avg')
    elif model_name == "resnet50":
        base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    elif model_name == "inceptionv3":
        base_model = InceptionV3(weights='imagenet', include_top=False, pooling='avg')
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    print(model_name)
    features = []
    for path in image_paths:
        img = Image.open(path).convert("RGB")
        print("IMAGES_SIZE_MODELS")
        print(IMAGES_SIZE_MODELS[model_name])
        img = img.resize((IMAGES_SIZE_MODELS[model_name],IMAGES_SIZE_MODELS[model_name]))  # Resize for ResNet
        img_array = np.array(img)
        preprocess = getPreprocess_input(model_name)
        img_array = preprocess(img_array)
        img_array = np.expand_dims(img_array, axis=0)
        feature = base_model.predict(img_array)
        features.append(feature.flatten())
    return np.array(features)

def apply_pca(train_features, test_features, n_components):
    print("Applying PCA")

   # Identify zero-variance features only from the training set
    zero_variance_features = np.where(np.std(train_features, axis=0) == 0)[0]

    # Remove zero-variance features from both train and test sets
    train_features = np.delete(train_features, zero_variance_features, axis=1)
    test_features = np.delete(test_features, zero_variance_features, axis=1)

    print(f"Number of zero-variance features BEFORE scaling: {len(zero_variance_features)}")

    # Standardize features
    scaler = StandardScaler()
    scale_train_features = scaler.fit_transform(train_features)
    scale_test_features = scaler.transform(test_features)

    print("Train mean:", np.mean(scale_train_features, axis=0)[:10])
    print("Train std:", np.std(scale_train_features, axis=0)[:10])

    # Apply PCA
    pca = PCA(n_components=n_components, svd_solver='randomized', random_state=42)
    train_features_pca = pca.fit_transform(scale_train_features)
    test_features_pca = pca.transform(scale_test_features)
    
    print("Components variance values")
    print(pca.explained_variance_ratio_)

    print("Cumulative variance")
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    print(cumulative_variance)

    # Find the number of components needed to retain 90% of variance
    n_components_90 = np.argmax(cumulative_variance >= 0.90) + 1
    print(f"Number of components for 90% variance: {n_components_90}")

    return train_features_pca, test_features_pca, pca, scaler

def residual_block(x, filters, stride=1):
    # Save the original input for the residual connection
    shortcut = x

    # First convolution layer
    x = layers.Conv2D(filters, kernel_size=3, strides=stride, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Second convolution layer
    x = layers.Conv2D(filters, kernel_size=3, strides=1, padding='same')(x)
    x = layers.BatchNormalization()(x)

    # Match the dimensions of the shortcut if needed
    if stride != 1 or x.shape[-1] != shortcut.shape[-1]:
        shortcut = layers.Conv2D(filters, kernel_size=1, strides=stride, padding='same')(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    # Add the shortcut to the output (residual connection)
    x = layers.add([x, shortcut])
    x = layers.ReLU()(x)

    return x


def ResNet18(input_shape=(224, 224, 3), weights="", include_top=False, pooling='avg'):
    inputs = layers.Input(shape=input_shape)
    
    # Initial Convolution and MaxPooling
    x = layers.Conv2D(64, kernel_size=7, strides=2, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(x)

    # Stage 1: 2 residual blocks with 64 filters
    x = residual_block(x, 64)
    x = residual_block(x, 64)

    # Stage 2: 2 residual blocks with 128 filters
    x = residual_block(x, 128, stride=2)  # Stride 2 for downsampling
    x = residual_block(x, 128)

    # Stage 3: 2 residual blocks with 256 filters
    x = residual_block(x, 256, stride=2)  # Stride 2 for downsampling
    x = residual_block(x, 256)

    # Stage 4: 2 residual blocks with 512 filters
    x = residual_block(x, 512, stride=2)  # Stride 2 for downsampling
    x = residual_block(x, 512)

    # Global Average Pooling (this will be the feature vector)
    x = layers.GlobalAveragePooling2D()(x)

    # The model will output the feature vector (without the classification layer)
    model = models.Model(inputs, x)

    return model

def ResNet30(input_shape=(224, 224, 3), weights="", include_top=False, pooling='avg'):
    inputs = layers.Input(shape=input_shape)
    
    # Initial Convolution and MaxPooling
    x = layers.Conv2D(64, kernel_size=7, strides=2, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(x)

    # Stage 1: 3 residual blocks with 64 filters
    x = residual_block(x, 64)
    x = residual_block(x, 64)
    x = residual_block(x, 64)

    # Stage 2: 4 residual blocks with 128 filters
    x = residual_block(x, 128, stride=2)  # Stride 2 for downsampling
    x = residual_block(x, 128)
    x = residual_block(x, 128)
    x = residual_block(x, 128)

    # Stage 3: 6 residual blocks with 256 filters
    x = residual_block(x, 256, stride=2)  # Stride 2 for downsampling
    x = residual_block(x, 256)
    x = residual_block(x, 256)
    x = residual_block(x, 256)
    x = residual_block(x, 256)
    x = residual_block(x, 256)

    # Stage 4: 3 residual blocks with 512 filters
    x = residual_block(x, 512, stride=2)  # Stride 2 for downsampling
    x = residual_block(x, 512)
    x = residual_block(x, 512)

    # Global Average Pooling (this will be the feature vector)
    x = layers.GlobalAveragePooling2D()(x)

    # The model will output the feature vector (without the classification layer)
    model = models.Model(inputs, x)

    return model