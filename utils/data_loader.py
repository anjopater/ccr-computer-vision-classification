# utils/data_loader.py
import os
import numpy as np
from PIL import Image
import logging
import imgaug.augmenters as iaa
import ssl
from sklearn.model_selection import GroupShuffleSplit, GridSearchCV, GroupKFold, KFold
from config import C_PATH, CCR_PATH, TEST_SIZE, RANDOM_STATE, BASE_PATH
import sys

# logging.basicConfig(
#     filename='app.log',  # Log file name
#     level=logging.INFO,  # Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
#     format='%(asctime)s - %(levelname)s - %(message)s'  # Log format
# )
# # Redirect print statements to the log file
# class LoggerWriter:
#     def __init__(self, logger):
#         self.logger = logger

#     def write(self, message):
#         if message.strip():  # Avoid logging empty lines
#             self.logger.info(message.strip())

#     def flush(self):
#         pass  # No need to implement for this example

# sys.stdout = LoggerWriter(logging)  # Redirect print() to logging


# Augmentation pipeline
augmentation_pipeline = iaa.Sequential([
    iaa.Fliplr(0.5),  # Horizontal flip
    iaa.Flipud(0.5),  # Vertical flip
    iaa.Affine(rotate=(-20, 20)),  # Rotation
    iaa.Multiply((0.8, 1.2)),  # Brightness adjustment
    iaa.GaussianBlur(sigma=(0.0, 1.0)),  # Blur
    iaa.MultiplySaturation((0.5, 1.5)),  # Saturation adjustment
    iaa.ElasticTransformation(alpha=(0, 5.0), sigma=0.25),  # Elastic deformations
    iaa.AdditiveGaussianNoise(scale=(0, 0.05 * 255)),  # Add Gaussian noise
])

def count_images_in_folder(folder):
    counts = {}
    for subfolder in os.listdir(folder):
        subfolder_path = os.path.join(folder, subfolder)
        if os.path.isdir(subfolder_path):
            images = [img for img in os.listdir(subfolder_path) if img.endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
            counts[subfolder] = len(images)
    return counts

def augment_images(folder, current_count, max_images):
    augmented_images = []
    images = [os.path.join(folder, img) for img in os.listdir(folder) if img.endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    while current_count + len(augmented_images) < max_images:
        img_path = np.random.choice(images)
        img = Image.open(img_path)
        img = np.array(img)
        aug_img = augmentation_pipeline(image=img)
        augmented_images.append(aug_img)
        save_path = os.path.join(folder, f"aug_{current_count + len(augmented_images)}.png")
        Image.fromarray(aug_img).save(save_path)
    logging.info(f"Completed augmentation in folder '{folder}'. Total images now: {max_images}")


def load_image_paths():
    """Load image paths and assign labels based on the numeric part of the filename."""
    image_paths = []
    labels = []
    
    # List all image files in the BASE_PATH directory
    for img_file in os.listdir(BASE_PATH):
        if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            image_paths.append(os.path.join(BASE_PATH, img_file))
            
            try:
                # Extract the numeric part of the filename (e.g., '0001.png' -> 1)
                image_index = int(img_file.split('.')[0])

                # Only process images with indices between 1 and 900
                if 1 <= image_index <= 900:
                    labels.append(image_index)  # Use the index as the class label
                else:
                    print(f"Skipping file {img_file} as its index is out of the expected range (1-900).")
            except ValueError:
                print(f"Skipping file: {img_file}. Cannot extract label from filename.")
    
    # Sort images by their numeric index to ensure proper sequence (e.g., 0001, 0002, 0900)
    sorted_image_paths_labels = sorted(zip(image_paths, labels), key=lambda x: int(os.path.basename(x[0]).split('.')[0]))
    
    # Unzip the sorted pairs
    sorted_image_paths, sorted_labels = zip(*sorted_image_paths_labels)
    
    # Return the sorted image paths and corresponding labels as numpy arrays
    return np.array(sorted_image_paths), np.array(sorted_labels)

def create_folds(image_paths, n_folds=3):
    """Creates folds by splitting 900 images (sorted by filename) into 3 folds.
    
    Each fold will contain 300 images: 10 classes per fold, each with 30 images.
    Class labels will be assigned from 1 to 10.
    """
    # Ensure n_folds is an integer
    n_folds = int(n_folds)
    
    # First, sort the image paths based on the numeric part of the filename.
    sorted_image_paths = sorted(
        image_paths, 
        key=lambda x: int(os.path.splitext(os.path.basename(x))[0])
    )
    
    folds = []
    # Each fold will contain 300 images.
    for fold_idx in range(n_folds):
        fold_images = []
        fold_labels = []
        for class_idx in range(10):  # 10 classes per fold
            # Compute the start and end indices for the current class in this fold.
            # For fold_idx = 0, indices go from 0 to 299.
            # Within that fold, for class_idx 0: images 0-29, class_idx 1: images 30-59, etc.
            start_idx = fold_idx * 300 + class_idx * 30
            end_idx = start_idx + 30
            fold_images.extend(sorted_image_paths[start_idx:end_idx])
            # Assign label as class_idx + 1 (to get labels 1 to 10)
            fold_labels.extend([class_idx + 1] * 30)
        
        folds.append((np.array(fold_images), np.array(fold_labels)))
    
    return folds



    # # Example call
    # image_paths, labels = load_image_paths()
    # folds = create_folds(image_paths, labels)

    # # Display fold information for verification
    # for fold_idx, (fold_images, fold_labels) in enumerate(folds):
    #     print(f"Fold {fold_idx + 1}:")
    #     unique_classes, class_counts = np.unique(fold_labels, return_counts=True)
    #     for class_id, count in zip(unique_classes, class_counts):
    #         print(f"  Class {class_id}: {count} images")

def load_data():
    image_paths, labels = load_image_paths()
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    
    train_images, test_images, train_labels, test_labels = [], [], [], []
    
    # Initialize lists for storing train and test data
    for fold, (train_idx, test_idx) in enumerate(kf.split(image_paths)):
        # Split images and labels into train and test sets for each fold
        train_images.append([image_paths[i] for i in train_idx])
        train_labels.append(labels[train_idx])
        test_images.append([image_paths[i] for i in test_idx])
        test_labels.append(labels[test_idx])

    # Return the train and test splits for all folds
    return train_images, train_labels, test_images, test_labels

# def load_data():
#     c_counts = count_images_in_folder(C_PATH)
#     ccr_counts = count_images_in_folder(CCR_PATH)
#     logging.info(f"C c_counts: {c_counts}")
#     logging.info(f"CR ccr_counts: {ccr_counts}")

#     max_images = max(list(c_counts.values()) + list(ccr_counts.values()))

#     # Apply augmentation
#     for animal, count in c_counts.items():
#         if count < max_images:
#             augment_images(os.path.join(C_PATH, animal), count, max_images)
#         else:
#             logging.info(f"No augmentation needed for folder '{animal}' in C group (image count: {count})")

#     for animal, count in ccr_counts.items():
#         if count < max_images:
#             augment_images(os.path.join(CCR_PATH, animal), count, max_images)
#         else:
#             logging.info(f"No augmentation needed for folder '{animal}' in CCR group (image count: {count})")

#     # Load image paths and labels
#     def get_image_paths_and_groups(base_path):
#         image_paths = []
#         groups = []
#         for animal in os.listdir(base_path):
#             animal_folder = os.path.join(base_path, animal)
#             if os.path.isdir(animal_folder):
#                 for img_file in os.listdir(animal_folder):
#                     if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
#                         image_paths.append(os.path.join(animal_folder, img_file))
#                         groups.append(animal)
#         return image_paths, groups

#     ssl._create_default_https_context = ssl._create_unverified_context
#     c_image_paths, c_groups = get_image_paths_and_groups(C_PATH)
#     ccr_image_paths, ccr_groups = get_image_paths_and_groups(CCR_PATH)

#     image_paths = c_image_paths + ccr_image_paths
#     groups = np.array(c_groups + ccr_groups)
#     labels = np.array([0] * len(c_image_paths) + [1] * len(ccr_image_paths))

#     # Split data using GroupShuffleSpli 0.2
#     gss = GroupShuffleSplit(test_size=0.2, train_size=0.8, n_splits=1, random_state=42)
#     train_idx, test_idx = next(gss.split(image_paths, labels, groups=groups))
#     train_images = [image_paths[i] for i in train_idx]
#     train_labels = labels[train_idx]
#     train_groups = groups[train_idx]
#     test_images = [image_paths[i] for i in test_idx]
#     test_labels = labels[test_idx]
#     test_groups = groups[test_idx]


#     # Verification step: Check if there is any overlap of animals between train and test
#     train_animals = set(train_groups)
#     test_animals = set(test_groups)

#     if train_animals & test_animals:  # If intersection is not empty, overlap exists
#         raise ValueError("Overlap between training and test animals detected!")
#     print("Data splited 80/20...")

#     # logging.info(f"Train animals: {train_groups}")
#     # logging.info(f"Test animals: {test_groups}")
#     print("Train animals...")
#     print(train_animals)
#     print("Test animals...")
#     print(test_animals)
#     print("Train groups...")
#     print(train_groups)
#     print("Test grouos...")
#     print(test_groups)
#     # print("Train images...")
#     # print(train_images)
#     # print("Test images...")
#     # print(test_images)
#     print("Test labels...")
#     print(test_labels)
#     print("Train labels...")
#     print(train_labels)
    

#     # print(train_images)
#     # print(test_images)

#     return train_images, train_labels, train_groups, test_images, test_labels, test_groups