"""
Jupyter Notebook – CNN fim-a-fim com load_data inline
----------------------------------------------------
Este notebook treina quatro backbones (ResNet-50, DenseNet-121,
EfficientNet-B0 e Inception-V3) em microfotografias de fígado.
"""

## 0. Instalações (execute apenas se necessário)
# !pip install tensorflow scikit-learn pillow imgaug --quiet

import os
import ssl
import logging
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers, regularizers
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.applications import (
    EfficientNetB0, DenseNet121, ResNet50, InceptionV3,
    efficientnet, densenet, resnet50, inception_v3
)
from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, confusion_matrix

## 2. Funções de coleta e split por animal
from typing import List, Tuple, Set, Dict

# configurações de logging e seed
data_home = os.getcwd()
signature = ssl._create_default_https_context
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("cnn_pipeline")

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

## 1. Variáveis e hiperparâmetros
BASE_PATH     = "/Users/antonio/Documents/projects/jupiterenv/datasets/HE_DATASET_LIVER"
CTRL_PATH     = os.path.join(BASE_PATH, "Controle")
CR_PATH       = os.path.join(BASE_PATH, "CRC")
EXTS          = (".png", ".jpg", ".jpeg", ".bmp")

BATCH_SIZE    = 32
EPOCHS_PHASE1 = 15
EPOCHS_PHASE2 = 60
K_FOLDS       = 4
CLASS_NAMES   = ["Controle", "CR"]

MODEL_CONFIG = {
    "resnet50":       {"input_size":224,  "preprocess":resnet50.preprocess_input,    "base":ResNet50,    "unfreeze":0.30},
    "densenet121":    {"input_size":224,  "preprocess":densenet.preprocess_input,     "base":DenseNet121, "unfreeze":0.10},
    "efficientnetb0": {"input_size":224,  "preprocess":efficientnet.preprocess_input, "base":EfficientNetB0,"unfreeze":0.25},
    "inceptionv3":    {"input_size":299,  "preprocess":inception_v3.preprocess_input,  "base":InceptionV3,  "unfreeze":0.25},
}


def _collect_paths_and_groups(
    base: str,
    prefix: str,
    exclude_word: str | None = None
) -> Tuple[List[str], List[str]]:
    paths, groups = [], []
    for animal in sorted(os.listdir(base)):
        animal_id = animal.lstrip("C")
        animal_dir = os.path.join(base, animal)
        if not os.path.isdir(animal_dir):
            continue
        for root, _, files in os.walk(animal_dir):
            for f in files:
                if not f.lower().endswith(EXTS):
                    continue
                if exclude_word and exclude_word.lower() in f.lower():
                    continue
                paths.append(os.path.join(root, f))
                groups.append(f"{prefix}_{animal_id}")
    return paths, groups

def count_images_in_folder(folder: str) -> Dict[str, int]:
    counts = {}
    for animal in sorted(os.listdir(folder)):
        path = os.path.join(folder, animal)
        if not os.path.isdir(path):
            continue
        n = 0
        for root, _, files in os.walk(path):
            n += sum(f.lower().endswith(EXTS) and not f.startswith("aug_") for f in files)
        counts[animal] = n
    return counts

def load_data():
    """
    Carrega todos os caminhos de imagem e realiza o split por animal:
      1) Conta quantas imagens originais cada animal tem em Controle e CRC.
      2) Coleta caminhos (paths) e rótulos de grupo (group IDs) para cada classe.
      3) Separa 2 animais Controle + 2 animais CRC (random seed fixa) como hold-out final.
      4) Constrói listas de treino e teste com base nesse hold-out.
      5) Exibe logs resumidos e devolve:
         - train_paths, train_labels, train_groups
         - test_paths, test_labels, test_groups
    """
    # 1) Contagem de imagens por animal
    ctl_counts = count_images_in_folder(CTRL_PATH)
    crc_counts = count_images_in_folder(CR_PATH)
    logger.info("Contagem Controle: %s", ctl_counts)
    logger.info("Contagem CRC     : %s", crc_counts)

    # 2) Coleta paths e grupos para as duas classes
    ctl_paths, ctl_groups = _collect_paths_and_groups(CTRL_PATH, "C")
    crc_paths, crc_groups = _collect_paths_and_groups(CR_PATH,  "CRC")

    all_paths  = ctl_paths + crc_paths
    all_groups = np.array(ctl_groups + crc_groups)
    all_labels = np.array([0]*len(ctl_paths) + [1]*len(crc_paths))

    # 3) Seleção fixa do hold-out: 2 Controle + 2 CRC
    rng = np.random.RandomState(SEED)
    ctrl_animals = np.unique(all_groups[all_labels == 0])
    crc_animals  = np.unique(all_groups[all_labels == 1])
    holdout_ctrl = rng.choice(ctrl_animals, size=2, replace=False)
    holdout_crc  = rng.choice(crc_animals,  size=2, replace=False)
    test_animals = set(np.concatenate([holdout_ctrl, holdout_crc]))

    # 4) Geração de índices de treino e teste
    test_idx  = [i for i, g in enumerate(all_groups) if g in test_animals]
    train_idx = [i for i, g in enumerate(all_groups) if g not in test_animals]

    train_paths  = [all_paths[i] for i in train_idx]
    test_paths   = [all_paths[i] for i in test_idx]
    train_labels = all_labels[train_idx]
    test_labels  = all_labels[test_idx]
    train_groups = all_groups[train_idx]
    test_groups  = all_groups[test_idx]

    # 5) Resumo dos splits
    logger.info("Animais treino: %s", sorted(set(train_groups)))
    logger.info("Animais teste : %s", sorted(set(test_groups)))
    logger.info("Distribuição treino: %s", np.bincount(train_labels))
    logger.info("Distribuição teste : %s", np.bincount(test_labels))

    return train_paths, train_labels, train_groups, test_paths, test_labels, test_groups


# carregando dados
train_paths, train_lbl, train_grp, test_paths, test_lbl, test_grp = load_data()


## 3. Converter paths em arrays pré-processados


def paths_to_array(paths: List[str], model_name: str) -> np.ndarray:
    cfg  = MODEL_CONFIG[model_name]
    size = cfg["input_size"]
    prep = cfg["preprocess"]
    X = []
    for p in paths:
        img = keras_image.load_img(p, target_size=(size, size))
        arr = keras_image.img_to_array(img)
        X.append(prep(arr))
    return np.array(X, dtype=np.float32)


## 4. Construção do modelo Keras


def build_model(model_name: str) -> models.Model:
    cfg  = MODEL_CONFIG[model_name]
    base = cfg["base"](
        weights="imagenet",
        include_top=False,
        input_shape=(cfg["input_size"], cfg["input_size"], 3),
        pooling="avg"
    )
    n_unfreeze = int(len(base.layers) * cfg["unfreeze"])
    for layer in base.layers[:-n_unfreeze]:
        layer.trainable = False
    for layer in base.layers[-n_unfreeze:]:
        layer.trainable = True
    head = models.Sequential([
        base,
        layers.BatchNormalization(),
        layers.Dense(256, activation="swish", kernel_regularizer=regularizers.l1_l2(1e-4,1e-3)),
        layers.Dropout(0.7),
        layers.BatchNormalization(),
        layers.Dense(64, activation="swish", kernel_regularizer=regularizers.l1_l2(1e-4,1e-3)),
        layers.Dropout(0.5),
        layers.Dense(1, activation="sigmoid", kernel_regularizer=regularizers.l2(1e-2))
    ])
    opt = optimizers.AdamW(learning_rate=1e-5, weight_decay=1e-4, clipnorm=0.5)
    head.compile(
        optimizer=opt,
        loss="binary_crossentropy",
        metrics=["accuracy",
                 tf.keras.metrics.AUC(name="auc"),
                 tf.keras.metrics.Precision(name="precision"),
                 tf.keras.metrics.Recall(name="recall")]
    )
    return head


## 5. Helpers: pesos de classe, CV e treino/validação em 2 fases


def get_class_weights(y: np.ndarray) -> Dict[int, float]:
    cw = class_weight.compute_class_weight("balanced", classes=np.unique(y), y=y)
    return dict(enumerate(cw))



def train_and_validate_cv(
    model_name: str,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray
) -> None:
    cv = StratifiedGroupKFold(n_splits=K_FOLDS, shuffle=True, random_state=SEED)
    for fold, (tr_idx, val_idx) in enumerate(cv.split(X, y, groups), 1):
        logger.info(f"[{model_name}] Fold {fold}/{K_FOLDS}")
        Xtr, ytr = X[tr_idx], y[tr_idx]
        Xval, yval = X[val_idx], y[val_idx]
        model = build_model(model_name)
        cw    = get_class_weights(ytr)
        # fase 1
        for layer in model.layers[0].layers:
            layer.trainable = False
        model.fit(
            Xtr, ytr,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS_PHASE1,
            validation_data=(Xval, yval),
            class_weight=cw,
            callbacks=[callbacks.EarlyStopping(patience=5, restore_best_weights=True)],
            verbose=2
        )
        # fase 2
        for layer in model.layers[0].layers:
            layer.trainable = True
        model.fit(
            Xtr, ytr,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS_PHASE2,
            validation_data=(Xval, yval),
            class_weight=cw,
            callbacks=[
                callbacks.ReduceLROnPlateau(patience=3, factor=0.2, min_lr=1e-7),
                callbacks.EarlyStopping(patience=7, restore_best_weights=True)
            ],
            verbose=2
        )
        m = model.evaluate(Xval, yval, verbose=0)
        logger.info(f"[{model_name}] Metrics: {dict(zip(model.metrics_names, m))}")


## 6. Treino final e teste hold-out


def final_train_and_test(
    model_name: str,
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xte: np.ndarray,
    yte: np.ndarray
) -> None:
    model = build_model(model_name)
    cw = get_class_weights(ytr)
    # fase 1
    for layer in model.layers[0].layers:
        layer.trainable = False
    model.fit(Xtr, ytr, batch_size=BATCH_SIZE, epochs=EPOCHS_PHASE1,
              class_weight=cw,
              callbacks=[callbacks.EarlyStopping(patience=5, restore_best_weights=True)],
              verbose=2)
    # fase 2
    for layer in model.layers[0].layers:
        layer.trainable = True
    model.fit(Xtr, ytr, batch_size=BATCH_SIZE, epochs=EPOCHS_PHASE2,
              class_weight=cw,
              callbacks=[
                  callbacks.ReduceLROnPlateau(patience=3, factor=0.2, min_lr=1e-7),
                  callbacks.EarlyStopping(patience=7, restore_best_weights=True)
              ],
              verbose=2)
    # avaliação
    metrics = model.evaluate(Xte, yte, verbose=0)
    preds   = (model.predict(Xte) > 0.5).astype(int)
    cr      = classification_report(yte, preds, target_names=CLASS_NAMES)
    cm      = confusion_matrix(yte, preds)
    logger.info(f"[{model_name}] Test metrics: {dict(zip(model.metrics_names, metrics))}")
    logger.info(f"Classification report:\n{cr}")
    logger.info(f"Confusion matrix:\n{cm}")


## 7. Loop principal para todos os backbones

for model_name in MODEL_CONFIG:
    Xtr   = paths_to_array(train_paths, model_name)
    Xte   = paths_to_array(test_paths,  model_name)
    ytr   = train_lbl
    yte   = test_lbl
    grp_tr= train_grp
    logger.info(f"=== Backbone: {model_name} ===")
    train_and_validate_cv(model_name, Xtr, ytr, grp_tr)
    final_train_and_test(model_name, Xtr, ytr, Xte, yte)
