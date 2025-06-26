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
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from config import IMAGES_SIZE_MODELS


from skimage import io, color, morphology, img_as_ubyte
from skimage.color import rgb2hed
from skimage.feature.texture import graycomatrix, graycoprops
from skimage.morphology import (
    opening, closing,
    area_opening, area_closing,
    reconstruction, disk
)

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
    elif model_name == "convNeXtTiny":
        base_model = ConvNeXtTiny(weights='imagenet', include_top=False, pooling='avg')
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


def compute_full_granulometry(
    image: np.ndarray,
    radii: list[int] = list(range(1, 51))) -> np.ndarray:
    """
    Compute multiple granulometry descriptors on a 2D uint8 image.
    Toggle each block by commenting/uncommenting; by default only BinAC is active.

    Descriptors:
      - GL Opening (gray-level opening)
      - GL Closing (gray-level closing)
      - BinAO    (binary area opening)
      - BinAC    (binary area closing) **default, as paper’s best**
      - GL SC    (gray-level structural closing)

    Returns
    -------
    feats : 1D float array
        Concatenated descriptors in the order listed above.
    """
    feats = []
    img = image.astype(np.float32)

    # 1) Gray‐level opening (GL Opening)
    # gl_open = []
    # for r in radii:
    #     gl_op = opening(img, disk(r)).astype(np.float32)
    #     gl_open.append(np.sum(img) - np.sum(gl_op))
    # feats.extend(gl_open)

    # 2) Gray‐level closing (GL Closing)
    # gl_close = []
    # for r in radii:
    #     gl_cl = closing(img, disk(r)).astype(np.float32)
    #     gl_close.append(np.sum(gl_cl) - np.sum(img))
    # feats.extend(gl_close)

    # 3) Binary area opening (BinAO)
    # bin_ao = []
    # mask = (image > 0)
    # for area in radii:
    #     ao = area_opening(mask, area_threshold=area)
    #     bin_ao.append(np.sum(mask) - np.sum(ao))
    # feats.extend(bin_ao)

    # 4) Binary area closing (BinAC)  <--- PAPER’S BEST
    bin_ac = []
    mask = (image > 0)
    for area in radii:
        ac = area_closing(mask, area_threshold=area)
        bin_ac.append(np.sum(ac))
    feats.extend(bin_ac)

    # 5) Gray‐level structural closing (GL SC)
    # gl_sc = []
    # for r in radii:
    #     sc = closing(img, disk(r)).astype(np.float32)
    #     gl_sc.append(np.sum(sc) - np.sum(img))
    # feats.extend(gl_sc)

    return np.array(feats, dtype=float)

# --------------------------------------------------------------
# Hand-crafted: Haralick texture + morphological granulometry
# --------------------------------------------------------------

# ───────────────────────────────────────────────────────────────────────────────
#  Low-level helpers (copiados do seu notebook) ────────────────────────────────
# ───────────────────────────────────────────────────────────────────────────────

from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage import (color, exposure, filters, measure, morphology, util,
                     feature, segmentation)
def hematoxylin_channel(rgb):
    """RGB → uint8 (0-255) onde núcleos são escuros/negativos → invertidos p/ brillantes."""
    rgb_f   = util.img_as_float(rgb[...,:3])
    h_raw   = color.rgb2hed(rgb_f)[..., 0]           # núcleos → negativo
    h_inv   = -h_raw                                 # núcleos → positivo
    low, hi = np.percentile(h_inv, (1, 99))
    h_clip  = np.clip(h_inv, low, hi)
    return exposure.rescale_intensity(
                h_clip, in_range=(low, hi), out_range=(0, 255)
           ).astype(np.uint8)

def smart_nuclei_mask(h, min_size=64):
    t    = filters.threshold_otsu(h)
    side = [h < t, h > t]           # escuro  vs  claro
    choose = max(side, key=lambda m: measure.label(
                 morphology.remove_small_objects(m, min_size)).max())
    mask = morphology.remove_small_objects(choose, min_size)
    mask = ndi.binary_fill_holes(mask)
    return mask

def tissue_mask(rgb, thresh=0.9):
    gray = color.rgb2gray(util.img_as_float(rgb))
    mask = gray < thresh
    return morphology.remove_small_holes(mask, area_threshold=10_000)

def split_touching(mask, min_distance=9):
    dist   = ndi.distance_transform_edt(mask)
    coords = feature.peak_local_max(dist, min_distance=min_distance, labels=mask)
    markers = np.zeros_like(mask, int)
    markers[tuple(coords.T)] = np.arange(1, coords.shape[0] + 1)
    labels  = segmentation.watershed(-dist, markers, mask=mask)
    return labels

def remove_giant(labels, max_size=7_000):
    areas = np.bincount(labels.ravel())
    too_big = np.where(areas > max_size)[0]
    for lbl in too_big:
        labels[labels == lbl] = 0
    return labels
# ───────────────────────────────────────────────────────────────────────────────


from skimage.morphology import (
    opening, closing,
    area_opening, area_closing,
    reconstruction, disk
)
import numpy as np

def compute_full_granulometry1(image: np.ndarray,
                              radii: list[int] = list(range(1, 51))
                             ) -> np.ndarray:
    """
    Compute the 6 granulometry signatures (structural, reconstruction,
    area) for both opening and closing, in gray‐level and binary form.

    Returns a 1D array of length 6 ops × 2 variants × len(radii).
    Order is:
      [Γ, Γᵦ, Γ_rec, Γᵦ,rec, Γ_area, Γᵦ,area,
       Φ, Φᵦ, Φ_rec, Φᵦ,rec, Φ_area, Φᵦ,area] each over radii.
    """
    img = image.astype(np.float32)
    feats = []

    # helper to binarize a residual
    def binarize(res):
        return (res > 0).astype(np.float32)

    # ---- OPENINGS ----
    prev = img.copy()
    for r in radii:
        selem = disk(r)

        # 1) Structural opening
        opened = opening(img, selem).astype(np.float32)
        resid = img - opened
        feats.append(resid.sum())               # Γ (gray)
        feats.append(binarize(resid).sum())     # Γᵦ (binary)

        # 2) Opening by reconstruction
        seed = opened
        rec = reconstruction(seed, img, method='dilation').astype(np.float32)
        resid_rec = img - rec
        feats.append(resid_rec.sum())           # Γ_rec
        feats.append(binarize(resid_rec).sum()) # Γᵦ,rec

        # 3) Area opening (area threshold = π·r²)
        area_thresh = np.pi * (r**2)
        aopen = area_opening(img, area_threshold=area_thresh).astype(np.float32)
        resid_area = img - aopen
        feats.append(resid_area.sum())          # Γ_area
        feats.append(binarize(resid_area).sum())# Γᵦ,area

    # ---- CLOSINGS ----
    for r in radii:
        selem = disk(r)

        # 4) Structural closing
        closed = closing(img, selem).astype(np.float32)
        resid = closed - img
        feats.append(resid.sum())               # Φ
        feats.append(binarize(resid).sum())     # Φᵦ

        # 5) Closing by reconstruction
        seed = closed
        rec = reconstruction(seed, img, method='erosion').astype(np.float32)
        resid_rec = rec - img
        feats.append(resid_rec.sum())           # Φ_rec
        feats.append(binarize(resid_rec).sum()) # Φᵦ,rec

        # 6) Area closing
        aclose = area_closing(img, area_threshold=np.pi*(r**2)).astype(np.float32)
        resid_area = aclose - img
        feats.append(resid_area.sum())          # Φ_area
        feats.append(binarize(resid_area).sum())# Φᵦ,area

    return np.array(feats, dtype=float)


def compute_granulometry(image: np.ndarray,
                         radii: list[int] = [1, 2, 4, 8, 16]) -> np.ndarray:
    """
    Compute a granulometry signature by successive openings.
    
    Parameters
    ----------
    image : 2D uint8
        Single‐channel image (e.g. hematoxylin channel) to analyze.
    radii : list of int
        Structuring element radii for opening.
    
    Returns
    -------
    1D array of float
        For each radius r, sum(prev_opened – current_opened),
        capturing how much “mass” is removed by that scale.
    """
    # convert to float so subtraction is safe
    img = image.astype(np.float32)
    prev = img.copy()
    feats = []
    for r in radii:
        selem = disk(r)
        opened = opening(img, selem).astype(np.float32)
        # how much area/intensity is removed by this opening
        removal = np.sum(prev) - np.sum(opened)
        feats.append(removal)
        prev = opened
    return np.array(feats, dtype=float)

def feature_extractor(rgb,
                      min_obj=30,
                      max_obj=7_000,
                      summary=True):
    """
    Segmenta núcleos em um tile RGB de H&E, devolve:
      • df_cells  – DataFrame com regionprops por núcleo
      • vec       – vetor de features agregadas (média, std, etc.)  (se summary=True)

    Parameters
    ----------
    rgb : ndarray uint8  (H×W×3)
    min_obj : int        (px²)  remove detritos menores
    max_obj : int        (px²)  descarta rótulos enormes (gordura / borda)
    summary : bool       gera ou não vetor slide-level

    Returns
    -------
    df_cells : pandas.DataFrame
    vec      : dict  |  None
    """
    # 1. H-channel  &  máscara esperta
    h      = hematoxylin_channel(rgb)
    mask   = smart_nuclei_mask(h, min_size=min_obj)
    mask  &= tissue_mask(rgb)

    # 2. Watershed + limpeza
    labels = split_touching(mask)
    labels = morphology.remove_small_objects(labels, min_obj)
    labels = remove_giant(labels, max_size=max_obj)
    labels[~tissue_mask(rgb)] = 0

    # 3. Regionprops por núcleo
    props = measure.regionprops_table(
                labels,
                intensity_image=h,
                properties=('area', 'eccentricity', 'solidity',
                            'major_axis_length', 'minor_axis_length',
                            'perimeter', 'mean_intensity', 'centroid')
            )
    df_cells = pd.DataFrame(props)

    if not summary:
        return df_cells, None

    # 4. Slide-level summary vector -------------------------------------------
    feat = {}
    for col in df_cells.columns:
        if col.startswith('centroid'):
            continue
        v = df_cells[col].values.astype(float)
        feat[f'{col}_mean'] = v.mean()
        feat[f'{col}_std']  = v.std(ddof=1)
        feat[f'{col}_p10']  = np.percentile(v, 10)
        feat[f'{col}_p90']  = np.percentile(v, 90)

    # densidade de núcleos
    tissue_px = tissue_mask(rgb).sum()
    feat['nuclei_per_1kpx'] = len(df_cells) / (tissue_px / 1_000 + 1e-6)

    # distância ao vizinho mais próximo
    if len(df_cells) >= 2:
        kd  = cKDTree(df_cells[['centroid-0', 'centroid-1']])
        nn  = kd.query(df_cells[['centroid-0','centroid-1']], k=2)[0][:,1]
        feat['nn_median'] = np.median(nn)
    else:
        feat['nn_median'] = np.nan

    return df_cells, feat


def extract_haralick_granulo(image_paths, radii=[1,2,4,8,16]):
    """
    For each image path:
      - load RGB
      - convert to HED
      - normalize each channel → uint8
      - compute Haralick + granulometry (toggleable)
      - stack into a single feature vector
    Returns an (N_images x N_features) array.
    """
    features = []

    def to_ubyte(chan: np.ndarray) -> np.ndarray:
        # min–max normalize float channel to [0,1], then to uint8
        c = chan.astype(np.float32)
        c = (c - c.min()) / (c.max() - c.min() + 1e-8)
        return img_as_ubyte(c)

    for path in image_paths:
        img = np.array(Image.open(path).convert("RGB"))
        hed = rgb2hed(img)

        # normalize each HED channel
        hemi  = to_ubyte(hed[..., 0])  # hematoxylin
        eosin = to_ubyte(hed[..., 1])  # eosin
        dab   = to_ubyte(hed[..., 2])  # DAB

           # --- Haralick on HEMI ---
        # glcm_he = graycomatrix(hemi,
        #                        distances=[1],
        #                        angles=[0],
        #                        levels=256,
        #                        symmetric=True,
        #                        normed=True)
        # har_hemi = [graycoprops(glcm_he, prop)[0,0]
                    # for prop in ("contrast","energy","homogeneity","correlation")]

        # --- Haralick on EOSIN (optional) ---
        glcm_eo = graycomatrix(eosin, distances=[1], angles=[0], levels=256,
                               symmetric=True, normed=True)
        har_eosin = [graycoprops(glcm_eo, p)[0,0]
                     for p in ("contrast","energy","homogeneity","correlation")]

        #--- Granulometry on HEMI ---
        # gran_hemi  = compute_full_granulometry(hemi,  radii=radii)

        # --- Granulometry on EOSIN  (optional) ---
        #gran_eosin = compute_granulometry(eosin, radii=radii)

        # --- Combine whichever you like ---
        # feat = np.hstack([har_hemi, har_eosin])
        # feat = np.hstack([har_hemi, har_eosin, gran_hemi, gran_eosin])


        # For now, just placeholder—uncomment above to use real features:
        
        # features.append(feat)
        
        _, nuclei_vec = feature_extractor(img, min_obj=30, max_obj=7000, summary=True)
        nuclei_vec = pd.Series(nuclei_vec).values    # dict → 1-D array
        vec = np.hstack([nuclei_vec, har_eosin]).astype(np.float32)
        features.append(vec)

    return np.vstack(features)

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