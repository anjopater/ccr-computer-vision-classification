# utils/data_loader.py
# ───────────────────────────────────────────────────────────────
import os, ssl, logging
import numpy as np
from PIL import Image
import imgaug.augmenters as iaa
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

from config import C_PATH, CCR_PATH, TEST_SIZE, RANDOM_STATE

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

EXTS = (".png", ".jpg", ".jpeg", ".bmp")

augmentation_pipeline = iaa.Sequential([
    iaa.Fliplr(0.5), iaa.Flipud(0.5),
    iaa.Affine(rotate=(-20, 20)),
    iaa.Multiply((0.8, 1.2)),
    iaa.GaussianBlur(sigma=(0.0, 1.0)),
    iaa.MultiplySaturation((0.5, 1.5)),
    iaa.ElasticTransformation(alpha=(0, 5.0), sigma=0.25),
    iaa.AdditiveGaussianNoise(scale=(0, 0.05 * 255)),
])

# ───────────────────────────────────────────────────────────────
def count_images_in_folder(folder: str) -> dict[str, int]:
    """Return {animal_id : #original_tiles} (aug_*.png excluded)."""
    counts = {}
    for animal in sorted(os.listdir(folder)):
        path = os.path.join(folder, animal)
        if not os.path.isdir(path):
            continue
        n = 0
        for root, _, files in os.walk(path):
            n += sum(
                f.lower().endswith(EXTS) and not f.startswith("aug_")
                for f in files
            )
        counts[animal] = n
    return counts

# ───────────────────────────────────────────────────────────────
def augment_images(animal_folder: str, curr: int, target: int) -> None:
    """Augment this animal until it has *target* images."""
    src = [
        os.path.join(root, f)
        for root, _, files in os.walk(animal_folder)
        for f in files
        if f.lower().endswith(EXTS) and not f.startswith("aug_")
    ]
    if not src:
        logging.warning("No source images in %s", animal_folder)
        return

    next_idx = curr + 1
    while curr < target:
        img_path = np.random.choice(src)
        with Image.open(img_path) as im:
            aug = augmentation_pipeline(image=np.array(im))

        save_dir = os.path.dirname(img_path)
        Image.fromarray(aug).save(os.path.join(save_dir, f"aug_{next_idx}.png"))

        curr += 1
        next_idx += 1
    logging.info("Augmented %s → %d files", animal_folder, target)

# ───────────────────────────────────────────────────────────────
# ───────────────────────────────────────────────────────────────
def _collect_paths_and_groups(
        base: str,
        prefix: str,
        exclude_word: str | None = None      # ← NOVO (ex.: "mask", "thumb")
    ):
    """
    Percorre todas as subpastas de *base* e devolve duas listas paralelas:
    • paths  – caminhos das imagens cujo sufixo está em EXTS
    • groups – rótulos de grupo (prefix + animal_id)

    Se *exclude_word* for passado, qualquer arquivo cujo nome contenha essa
    palavra (case-insensitive) será ignorado.
    """
    paths, groups = [], []

    for animal in os.listdir(base):
        animal_id = animal.lstrip("C")       # "C1" -> "1", "1" fica "1"

        for root, _, files in os.walk(os.path.join(base, animal)):
            for f in files:
                fname = f.lower()

                # ——— filtro de extensão ———
                if not fname.endswith(EXTS):
                    continue

                # ——— filtro de exclusão ———
                if exclude_word and exclude_word.lower() not in fname:
                    continue

                print(fname)
                paths.append(os.path.join(root, f))
                groups.append(f"{prefix}_{animal_id}")

    return paths, groups


# ───────────────────────────────────────────────────────────────
def load_data():
    # 1) count original tiles
    ctl_counts = count_images_in_folder(C_PATH)
    crc_counts = count_images_in_folder(CCR_PATH)
    logging.info("Control counts : %s", ctl_counts)
    logging.info("CRC counts     : %s", crc_counts)

    max_tiles = max((*ctl_counts.values(), *crc_counts.values()))

    # 2) augment up to max_tiles
    # for a, n in ctl_counts.items():
    #     if n < max_tiles:
    #         augment_images(os.path.join(C_PATH, a), n, max_tiles)
    # for a, n in crc_counts.items():
    #     if n < max_tiles:
    #         augment_images(os.path.join(CCR_PATH, a), n, max_tiles)

    # 3) collect paths / labels / groups
    ctl_paths, ctl_groups = _collect_paths_and_groups(C_PATH,  "C", "_")
    crc_paths, crc_groups = _collect_paths_and_groups(CCR_PATH, "CRC","_")

    paths   = ctl_paths + crc_paths
    groups  = np.array(ctl_groups + crc_groups)
    labels  = np.array([0]*len(ctl_paths) + [1]*len(crc_paths))

    # 4) mixed-class hold-out split (25 %)
    
# 4) mixed-class hold-out split (25%)
    gss = GroupShuffleSplit(test_size=0.25, n_splits=1, random_state=42)
    for tr_idx, te_idx in gss.split(paths, labels, groups):
        # stop as soon as the test set has both classes
        if len(np.unique(labels[te_idx])) == 2:
            break
    else:
        raise RuntimeError("Could not draw mixed-class test set")
    
    # cv_holdout = GroupShuffleSplit(test_size=0.25, n_splits=1, random_state=42)
    # for tr_idx, te_idx in gss.split(paths, labels, groups):
    #     if len(np.unique(labels[te_idx])) == 2:
    #         break
    # else:
    #     raise RuntimeError("Could not draw mixed-class test set")

    # tr_idx, te_idx = next(
    #     cv_holdout.split(paths, labels, groups)
    # )
    
    RNG = np.random.RandomState(42)

    ctrl_animals = np.unique(groups[labels == 0])
    crc_animals  = np.unique(groups[labels == 1])

    # force exactly 1 control + 3 CRC
    test_ctrl = RNG.choice(ctrl_animals, size=2, replace=False)
    test_crc  = RNG.choice(crc_animals,  size=2, replace=False)

    test_groups = set(np.concatenate([test_ctrl, test_crc]))

    test_idx  = [i for i, g in enumerate(groups) if g in test_groups]
    train_idx = [i for i in range(len(groups)) if i not in test_idx]
    

    # # 5) organise outputs
    # train_paths  = [paths[i] for i in tr_idx]
    # test_paths   = [paths[i] for i in te_idx]
    # train_groups = groups[tr_idx]
    # test_groups  = groups[te_idx]
    # train_labels = labels[tr_idx]
    # test_labels  = labels[te_idx]
    
    train_paths  = [paths[i] for i in train_idx]
    test_paths   = [paths[i] for i in test_idx]
    train_labels = labels[train_idx]
    test_labels  = labels[test_idx]
    train_groups = groups[train_idx]
    test_groups  = groups[test_idx]

    # 6) log split summary
    logging.info("Train animals : %s", sorted(set(train_groups)))
    logging.info("Test  animals : %s", sorted(set(test_groups)))
    logging.info("Train label counts : %s", np.bincount(train_labels))
    logging.info("Test  label counts : %s", np.bincount(test_labels))

    return (train_paths,  train_labels, train_groups,
            test_paths,   test_labels,  test_groups)
