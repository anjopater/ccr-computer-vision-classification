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
import pywt
from scipy.stats import skew, kurtosis
from tensorflow.keras import layers, models
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import VarianceThreshold
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from config import IMAGES_SIZE_MODELS

import cv2                           # only for RGB→gray conversion
from scipy.signal import convolve2d
from skimage.feature import local_binary_pattern

from skimage import io, color, morphology, img_as_ubyte
from skimage.color import rgb2hed
from skimage.feature.texture import graycomatrix, graycoprops
from skimage.morphology import (
    opening, closing,
    area_opening, area_closing,
    reconstruction, disk
)

from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage import (color, exposure, filters, measure, morphology, util,
                     feature, segmentation)

from skimage.morphology import (
    opening, closing,
    area_opening, area_closing,
    reconstruction, disk
)
import numpy as np

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

# ────────────────────────────────────────────────────────────────
# 1)  LPQ  — Local Phase Quantization
# ----------------------------------------------------------------
def extract_lpq_features(img, win_size: int = 7, freq: float = 1.0) -> np.ndarray:
    """
    Compute LPQ histogram over the entire image.
    
    Parameters
    ----------
    img : ndarray
        Grayscale (H×W) or RGB/BGR (H×W×3).
    win_size : int
        Local window size (odd). 7 or 9 are common in texture work.
    freq : float
        Central frequency of the short 2-D DFT basis, usually 1.0.

    Returns
    -------
    hist : ndarray, shape (256,)
        Normalised LPQ code histogram (8-bit code → 256 bins).
    """
    # --- 1. preprocess --------------------------------------------------------
    if img.ndim == 3:                           # RGB → gray
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = img.astype(np.float32)
    r    = win_size // 2                        # radius
    x    = np.arange(-r, r + 1)

    # --- 2. build 4 short DFT basis filters (as in Ojansivu & Heikkilä, 2008) -
    w0   = np.exp(-2j * np.pi * x * freq / win_size)
    W    = np.stack([np.real(w0), np.imag(w0)], axis=0)  # shape (2,win_size)
    
    # Vertical & horizontal separable filters
    filters = [
        np.outer(W[0], np.ones_like(x)),        # 1) Re{F(0,ω)}
        np.outer(W[1], np.ones_like(x)),        # 2) Im{F(0,ω)}
        np.outer(np.ones_like(x), W[0]),        # 3) Re{F(ω,0)}
        np.outer(np.ones_like(x), W[1])         # 4) Im{F(ω,0)}
    ]

    # --- 3. filter responses per pixel ---------------------------------------
    responses = [convolve2d(img, f, mode='same', boundary='symm') for f in filters]
    responses = np.stack(responses, axis=-1)    # (H,W,4)

    # --- 4. decorrelate & binarise (sign bit for each channel) ---------------
    # Simple whitening: subtract mean and divide by std per channel
    resp   = (responses - responses.mean(axis=(0,1))) / (responses.std(axis=(0,1)) + 1e-8)
    codes  = (resp > 0).astype(np.uint8)
    
    # Pack 4 binary planes into 8-bit code (2^4=16 possible values) → extend
    code_img = (codes[...,0] << 3) | (codes[...,1] << 2) | \
               (codes[...,2] << 1) |  codes[...,3]

    # --- 5. histogram normalised --------------------------------------------
    hist, _ = np.histogram(code_img, bins=256, range=(0,255), density=False)
    hist    = hist.astype(np.float32)
    hist   /= hist.sum() + 1e-12               # L1-normalise

    return hist                                # shape (256,)

# ────────────────────────────────────────────────────────────────
# 2)  LBP  — Local Binary Pattern
# ----------------------------------------------------------------
def extract_lbp_features(
        img,
        radius: int = 6,
        n_points: int = None,
        n_bins:   int = None,
        method:   str = "uniform") -> np.ndarray:
    """
    Extract an LBP histogram over the whole image.

    Parameters
    ----------
    img : ndarray
        Grayscale (H×W) or RGB (H×W×3).
    radius : int
        Pixel radius around the centre (1 → 8 neighbours).
    n_points : int or None
        Number of sampling points. Default = 8 * radius.
    n_bins : int or None
        Histogram length. If None, picks (n_points + 2) for 'uniform' and
        2**n_points otherwise.
    method : str
        LBP variant: 'uniform', 'default', 'ror', 'var'.

    Returns
    -------
    hist : ndarray
        Normalised LBP histogram (length = n_bins).
    """
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.astype(np.uint8)

    if n_points is None:
        n_points = 8 * radius
    lbp = local_binary_pattern(img, n_points, radius, method)

    if n_bins is None:
        n_bins = n_points + 2 if method == "uniform" else 2 ** n_points

    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins-1))
    hist = hist.astype(np.float32)
    hist /= hist.sum() + 1e-12                 # L1 normalise
    return hist

def extract_wavelet_features(
        img,
        wavelet: str = "db4",
        levels: int = 3,
        stats: tuple = ("mean", "std", "energy", "entropy", "skew", "kurtosis")
    ) -> np.ndarray:
    """
    Extracts summary statistics from each sub-banda da DWT.

    Parameters
    ----------
    img : ndarray
        Grayscale image (H×W) ou RGB (H×W×3).  Se RGB, o cálculo
        é feito canal-a-canal e depois concatenado.
    wavelet : str
        Nome do wavelet mãe (ex.: "db4", "sym4", "haar"...).
    levels : int
        Profundidade da decomposição. 2-3 já costuma bastar.
    stats : tuple[str]
        Quais estatísticas calcular em cada sub-banda.
        Opções disponíveis: "mean", "std", "energy", "entropy",
        "skew", "kurtosis".

    Returns
    -------
    features : 1-D ndarray
        Vetor (float64) com len(stats) × (1 + 3×levels) × n_channels
        elementos.
    """
    
    # —— helpers ————————————————————————————————
    def _band_stats(band):
        band = band.astype(np.float64)
        r = band.ravel()
        res = []
        if "mean" in stats: res.append(np.mean(r))
        if "std" in stats:  res.append(np.std(r))
        if "energy" in stats:
            # Use log1p, que calcula log(1 + x) para estabilidade numérica
            energy_val = np.sum(r**2)
            res.append(np.log1p(energy_val))
        if "entropy" in stats:
            p = np.abs(r)
            p = p / (p.sum() + 1e-12)
            res.append(-np.sum(p * np.log(p + 1e-12)))
        if "skew" in stats:
            if np.std(r) == 0 or np.isnan(r).any():
                res.append(0.0)
            else:
                res.append(skew(r))
        if "kurtosis" in stats:
            if np.std(r) == 0 or np.isnan(r).any():
                res.append(0.0)
            else:
                res.append(kurtosis(r))
        return res

    # —— garante forma (H,W,C) ————————————————————
    img = img.astype(np.float32)
    if img.ndim == 2:          # grayscale → (H,W,1)
        img = img[..., None]

    feats = []
    for c in range(img.shape[2]):
        coeffs = pywt.wavedec2(img[..., c], wavelet=wavelet, level=levels)
        # LL
        feats.extend(_band_stats(coeffs[0]))
        # detalhes LH, HL, HH em cada nível
        for detail in coeffs[1:]:
            for band in detail:
                feats.extend(_band_stats(band))
                
    # feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)

    return np.array(feats, dtype=np.float64)

def extract_handcrafted_features(image_paths, radii=[1,2,4,8,16],  target_size=(224, 224)):
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
        #img = np.array(Image.open(path).convert("RGB"))
        img = Image.open(path).convert("RGB")

        print(path)
        # # --- CHANGE 2: Add the resizing step here ---
        # if target_size:
        #     img = img.resize(target_size, Image.Resampling.LANCZOS)
        
        img = np.array(img)
        
        hed = rgb2hed(img)

        # normalize each HED channel
        hemi  = to_ubyte(hed[..., 0])  # hematoxylin
        eosin = to_ubyte(hed[..., 1])  # eosin
        dab   = to_ubyte(hed[..., 2])  # DAB

           # --- Haralick on HEMI ---
        glcm_he = graycomatrix(hemi,
                                distances=[1,2,4],
                                angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                               levels=256,
                               symmetric=True,
                               normed=True)
        props = np.array([
            graycoprops(glcm_he, p).mean()      # .mean() já média tudo
            for p in ("contrast", "energy", "homogeneity", "correlation")
        ], dtype=np.float32)
        har_hemi = [graycoprops(glcm_he, prop)[0,0]
                    for prop in ("contrast","energy","homogeneity","correlation")]

        # --- Haralick on EOSIN (optional) ---
        glcm_eo = graycomatrix(eosin, distances=[1], angles=[0], levels=256,
                               symmetric=True, normed=True)
        har_eosin = [graycoprops(glcm_eo, p).mean() 
                     for p in ("contrast","energy","homogeneity","correlation")]
        
        # lpb = extract_lbp_features(hemi)
        
        w_vector = extract_wavelet_features(hemi, wavelet="db5", levels=3)

        vec = np.hstack([w_vector]).astype(np.float32)
        # vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)

        features.append(vec)

    return np.vstack(features)
