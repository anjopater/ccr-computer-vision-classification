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
from tensorflow.keras.applications.convnext import preprocess_input as convnext_preprocess

from tensorflow.keras import layers, models
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import VarianceThreshold

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from config import IMAGES_SIZE_MODELS

import numpy as np
from skimage import io, color, feature, filters, measure
from skimage.filters import sobel
from skimage.color import rgb2gray
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.util import img_as_ubyte
from PIL import Image
from histomicstk.preprocessing.color_deconvolution import separate_stains_xu_snmf
import skimage.io as io
from skimage.color import rgb2hed
from histomicstk.preprocessing.color_deconvolution import color_deconvolution
from skimage.feature import graycomatrix, graycoprops
from skimage.util import img_as_ubyte
from skimage import color, io, feature, measure

def getPreprocess_input(model_name):
    if model_name == "resnet50":
        return resnet_preprocess
    elif model_name == "efficientNetB0":
        return efficientnet_preprocess
    elif model_name == "densenet121":
        return densenet_preprocess
    elif model_name == "inceptionv3":
        return inception_preprocess
    elif model_name == "convNeXtTiny":
        return convnext_preprocess
    
# def extract_cnn_features(image_paths, model_name):
#     print("EXTRATING FEATURES WITH MODEL")
#     print(model_name == "resnet50")

#     if model_name == "efficientNetB0":
#         base_model = EfficientNetB0(weights='imagenet', include_top=False, pooling='avg')
#     elif model_name == "densenet121":
#         base_model = DenseNet121(weights='imagenet', include_top=False, pooling='avg')
#     elif model_name == "resnet50":
#         base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
#     elif model_name == "inceptionv3":
#         base_model = InceptionV3(weights='imagenet', include_top=False, pooling='avg')
#     elif model_name == "convNeXtTiny":
#         base_model = ConvNeXtTiny(weights='imagenet', include_top=False, pooling='avg')
#     else:
#         raise ValueError(f"Unsupported model: {model_name}")
#     print(model_name)
#     features = []
#     for path in image_paths:
#         img = Image.open(path).convert("RGB")
#         print("IMAGES_SIZE_MODELS")
#         print(IMAGES_SIZE_MODELS[model_name])
#         img = img.resize((IMAGES_SIZE_MODELS[model_name],IMAGES_SIZE_MODELS[model_name]))  # Resize for ResNet
#         img_array = np.array(img)
#         preprocess = getPreprocess_input(model_name)
#         img_array = preprocess(img_array)
#         img_array = np.expand_dims(img_array, axis=0)
#         feature = base_model.predict(img_array)
#         features.append(feature.flatten())
#     return np.array(features)

def apply_pca(train_features, test_features, n_components):
    print("Applying PCA")

    # Remove zero-variance features (fit only on train, transform both)
    selector = VarianceThreshold(threshold=0.0)
    train_features = selector.fit_transform(train_features)
    test_features = selector.transform(test_features)

    print(f"Number of zero-variance features removed: {train_features.shape[1] - test_features.shape[1]}")

    # Standardize features (fit only on train, transform both)
    scaler = StandardScaler()
    scale_train_features = scaler.fit_transform(train_features)
    scale_test_features = scaler.transform(test_features)

    print("Train mean:", np.mean(scale_train_features, axis=0)[:10])
    print("Train std:", np.std(scale_train_features, axis=0)[:10])

    # Apply PCA
    pca = PCA(n_components=n_components, svd_solver='auto', random_state=42)
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

# Function to extract LBP features
def extract_lbp(image):
    gray_img = rgb2gray(image)
    radius = 3
    n_points = 8 * radius
    lbp = local_binary_pattern(gray_img, n_points, radius, method="uniform")
    hist, _ = np.histogram(lbp, bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)  # Normalize the histogram
    return hist

# Function to extract PAS-specific features (e.g., color deconvolution or staining intensity)
def extract_pas_features(image, stain_matrix, display_steps=False):
    """
    Extracts PAS-specific features using IHC color deconvolution.
    
    Parameters:
        image (numpy.ndarray): Input RGB image.
        stain_matrix (numpy.ndarray): Stain vectors for deconvolution.
        display_steps (bool): If True, displays intermediate steps.
    
    Returns:
        numpy.ndarray: Feature vector containing PAS intensity, contrast, and mean area.
    """
    # Step 1: Perform color deconvolution
    deconvolved = color_deconvolution(image, stain_matrix)
    pas_channel = deconvolved.Stains[:, :, 0]  # PAS channel (first stain)
    background_channel = deconvolved.Stains[:, :, 1]  # Background channel (second stain)
    
    # if display_steps:
    #     plt.figure(figsize=(15, 5))
    #     plt.subplot(1, 3, 1)
    #     plt.imshow(image)
    #     plt.title("Original Image")
    #     plt.axis('off')
        
    #     plt.subplot(1, 3, 2)
    #     plt.imshow(pas_channel, cmap='gray')
    #     plt.title("PAS Channel")
    #     plt.axis('off')
        
    #     plt.subplot(1, 3, 3)
    #     plt.imshow(background_channel, cmap='gray')
    #     plt.title("Background Channel")
    #     plt.axis('off')
        
    #     plt.show()
    
    # Step 2: Threshold the PAS channel to create a binary mask
    pas_mask = pas_channel > 0.5  # Adjust threshold as needed
    
    # Check if pas_mask is empty
    if np.sum(pas_mask) == 0:
        print("Warning: No PAS-positive regions found. Adjust thresholds or check the image.")
        return np.array([0.0, 0.0, 0.0])  # Return default values
    
    # Step 3: Compute PAS intensity
    pas_intensity = np.mean(pas_channel[pas_mask])
    
    # Step 4: Compute texture features (e.g., contrast)
    pas_channel_uint8 = img_as_ubyte(pas_channel)
    glcm = graycomatrix(pas_channel_uint8, distances=[1], angles=[0], symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    
    # Step 5: Compute morphological features (e.g., mean area of PAS-positive regions)
    labeled_image = measure.label(pas_mask)
    regions = measure.regionprops(labeled_image)
    areas = [region.area for region in regions]
    mean_area = np.mean(areas) if areas else 0
    
    # Return feature vector
    return np.array([pas_intensity, contrast, mean_area])

# Function to extract GLCM texture features
def extract_glcm_features(image):
    gray_img = rgb2gray(image)
    glcm = graycomatrix(img_as_ubyte(gray_img), distances=[1], angles=[0], symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    return np.array([contrast, correlation, energy, homogeneity])

# Function to extract edge features using Sobel filter
def extract_edge_features(image):
    gray_img = rgb2gray(image)
    edges = sobel(gray_img)
    edge_mean = np.mean(edges)
    edge_std = np.std(edges)
    return np.array([edge_mean, edge_std])

# Function to extract color features (mean and std of each channel)
def extract_color_features(image):
    mean_r, mean_g, mean_b = np.mean(image, axis=(0, 1))
    std_r, std_g, std_b = np.std(image, axis=(0, 1))
    return np.array([mean_r, mean_g, mean_b, std_r, std_g, std_b])

# Main function to extract all features
def extract_traditional_features(image_paths):
    print("EXTRACTING FEATURES WITH TRADITIONAL METHODS")


    stain_matrix = np.array([
        [0.47274575, 0.3625579,  0.80315828],
    [0.13392691, 0.76943798, 0.62452284],
    [0.92287211, 0.38201481, 0.04870053]
    ])
    
    features = []
    for path in image_paths:
        img = Image.open(path).convert("RGB")
        img = img.resize((224, 224))  # Resize to a common size (e.g., 224x224)
        img_array = np.array(img)
        
        # Extract features from each feature extractor
        #lbp_features = extract_lbp(img_array)
        # pas_features = extract_pas_features(img_array, stain_matrix)
        #glcm_features = extract_glcm_features(img_array)
        edge_features = extract_edge_features(img_array)
        #color_features = extract_color_features(img_array)
        
        # Concatenate all features into a single feature vector
        feature_vector = np.concatenate([
            # lbp_features,
            # pas_features,
            # glcm_features,
            edge_features,
            # color_features
        ])
        
        features.append(feature_vector)
    
    return np.array(features)