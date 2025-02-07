# utils/evaluator.py
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import numpy as np
import warnings
from sklearn.exceptions import ConvergenceWarning
from utils.save_plots import plot_and_save_confusion_matrix, plot_custom_cv, plot_and_save_pca, plot_and_save_umap
from sklearn.decomposition import PCA
import umap

from sklearn.preprocessing import StandardScaler
import os

def get_classifiers():
    return {
        'KNN': {
            'model': KNeighborsClassifier(),
            'params': {
                'n_neighbors': [1, 3, 5, 7, 9],
                'weights': ['uniform', 'distance'],
                'p': [1, 2]
            }
        },
        'SVM': {
            'model': SVC(probability=True),
            'params': {
                'C': [0.1, 1, 10],
                'kernel': ['linear', 'rbf'],
                'gamma': ['scale', 'auto']
            }
        },
        'Random Forest': {
            'model': RandomForestClassifier(),
            'params': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            }
        },
        'Logistic Regression': {
            'model': LogisticRegression(max_iter=500),
            'params': {
                'C': [0.1, 1, 10],
                'penalty': ['l2', 'none'],
                'solver': ['lbfgs', 'saga']
            }
        },
        'MLP': {
            'model': MLPClassifier(),
            'params': {
                'hidden_layer_sizes': [(128, 64, 32)],
                'activation': ['relu', 'logistic', 'tanh', 'identity'],
                'solver': ['adam', 'sgd', 'lbfgs'],
                'learning_rate': ['constant', 'adaptive', 'invscaling'],
                'max_iter': [200, 500, 1000],
                'alpha': [0.0001, 0.001, 0.01]  # Regularization strength
            }
        },
        'MLP2': {
            'model': MLPClassifier(),
            'params': {
                'hidden_layer_sizes': [(64,), (64, 32)],
                'activation': ['relu', 'logistic', 'tanh', 'identity'],
                'solver': ['adam', 'sgd', 'lbfgs'],
                'learning_rate': ['constant', 'adaptive', 'invscaling'],
                'max_iter': [200, 500, 1000],
                'alpha': [0.0001, 0.001, 0.01]  # Regularization strength
            }
        },
        # 'XGBoost': {
        #     'model': xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
        #     'params': {
        #         'n_estimators': [50, 100, 200],
        #         'learning_rate': [0.01, 0.1, 0.2],
        #         'max_depth': [3, 5, 7],
        #         'subsample': [0.7, 0.8, 1.0]
        #     }
        # }
    }

# def cross_validation_with_pca(extracted_features, n_components):
#     """
#     Performs cross-validation using PCA and a classifier on the provided folds.
    
#     Parameters:
#       extracted_features: list of tuples, where each tuple is (features, labels) for a fold.
#                           Each fold has 300 images (10 classes × 30 images).
#       n_components: number of PCA components.
#     """
#     from sklearn.decomposition import PCA
#     from sklearn.svm import SVC
#     import numpy as np
#     print(extracted_features)
#     n_folds = len(extracted_features)
#     fold_accuracies = []
#     pcas = []
    
#     # Leave-one-fold-out cross-validation:
#     for test_fold_idx in range(n_folds):
#         print(f"Processing fold {test_fold_idx + 1} as test fold...")
        
#         # Use the test_fold_idx as test, and combine the rest as training
#         test_features, test_labels = extracted_features[test_fold_idx]
#         train_features_list = []
#         train_labels_list = []
#         for i in range(n_folds):
#             if i != test_fold_idx:
#                 features, labels = extracted_features[i]
#                 train_features_list.append(features)
#                 train_labels_list.append(labels)
                
#         # Concatenate training features and labels from the other folds
#         train_features = np.concatenate(train_features_list, axis=0)
#         train_labels = np.concatenate(train_labels_list, axis=0)
        
#         # Fit PCA on the training set only
#         pca = PCA(n_components=n_components)
#         train_features_pca = pca.fit_transform(train_features)
#         test_features_pca = pca.transform(test_features)
#         pcas.append(train_features_pca)
#         # Train a classifier on the PCA-transformed training data
#         # classifier = SVC(kernel='linear', C=00.1, gamma="scale")
#         # test_features_pca = pca.transform(test_features)

        
             
#         classifier = LogisticRegression(max_iter=500, C=0.1, penalty="l2", solver="saga" )
#         classifier.fit(train_features_pca, train_labels)
#         # Evaluate the classifier on the PCA-transformed test data
#         accuracy = classifier.score(test_features_pca, test_labels)
#         print(f"Accuracy for fold {test_fold_idx + 1}: {accuracy * 100:.2f}%")
#         fold_accuracies.append(accuracy)
#     avg_accuracy = np.mean(fold_accuracies)
#     print(f"Average cross-validation accuracy: {avg_accuracy * 100:.2f}%")
#     plot_custom_cv(extracted_features)
    


def cross_validation_with_pca(extracted_features, n_components):
    """
    Performs cross-validation using PCA and UMAP for visualization and a classifier on the provided folds.
    
    Parameters:
      extracted_features: list of tuples, where each tuple is (features, labels) for a fold.
                          Each fold has 300 images (10 classes × 30 images).
      n_components: number of components for UMAP and PCA.
    """
    n_folds = len(extracted_features)
    fold_accuracies = []

    all_true_labels = []  # Store all true labels
    all_pred_labels = []  # Store all predicted labels
    all_features = []     # Store all features for final UMAP plot
    all_labels = []       # Store all labels for final UMAP plot

    for test_fold_idx in range(n_folds):
        print(f"Processing fold {test_fold_idx + 1} as test fold...")
        
        test_features, test_labels = extracted_features[test_fold_idx]
        train_features_list, train_labels_list = [], []
        
        for i in range(n_folds):
            if i != test_fold_idx:
                features, labels = extracted_features[i]
                train_features_list.append(features)
                train_labels_list.append(labels)
                
        train_features = np.concatenate(train_features_list, axis=0)
        train_labels = np.concatenate(train_labels_list, axis=0)
        
        # Apply PCA to the training set
        pca = PCA(n_components=60)
        train_features_pca = pca.fit_transform(train_features)
        test_features_pca = pca.transform(test_features)

        # Store for final PCA plot
        all_features.append(train_features_pca)  # Store PCA-transformed training data
        all_labels.append(train_labels)

        # Train classifier
        classifier = LogisticRegression(max_iter=500, C=0.1, penalty="l2", solver="saga")
        classifier.fit(train_features_pca, train_labels)
        
        # Predict on test set
        y_pred = classifier.predict(test_features_pca)
        accuracy = classifier.score(test_features_pca, test_labels)
        
        print(f"Accuracy for fold {test_fold_idx + 1}: {accuracy * 100:.2f}%")
        fold_accuracies.append(accuracy)

        # Store for final confusion matrix
        all_true_labels.extend(test_labels)
        all_pred_labels.extend(y_pred)

    # Compute average accuracy
    avg_accuracy = np.mean(fold_accuracies)
    print(f"Average cross-validation accuracy: {avg_accuracy * 100:.2f}%")

    # Concatenate the PCA-transformed data from all folds into a 2D array
    all_features_pca = np.concatenate(all_features, axis=0)  # Concatenate PCA features
    all_labels = np.concatenate(all_labels, axis=0)  # Concatenate labels
    
    output_dir = os.path.join("results", "ResNet", f"pca_{n_components}")
    os.makedirs(output_dir, exist_ok=True)

    # Save PCA plot for the entire dataset
    plot_and_save_pca(
        all_features_pca, all_labels, 
        title=f"PCA for n_components={3}", 
        filename="full_pca_plot.png", 
        output_dir=output_dir
    )

    # Apply UMAP for dimensionality reduction
    umap_model = umap.UMAP(n_components=2)  # Set n_components to 2 for 2D visualization
    all_features_umap = umap_model.fit_transform(all_features_pca)  # Apply UMAP to PCA features

    output_dir = os.path.join("results", "ResNet", f"umap_{n_components}")
    os.makedirs(output_dir, exist_ok=True)

    # Save UMAP plot for visualization
    plot_and_save_umap(
        features_umap= all_features_umap,
        labels=all_labels, 
        title=f"UMAP for n_components={2}",  # 2D UMAP for plotting
        filename="full_umap_plot.png", 
        output_dir=output_dir
)

    # Generate final confusion matrix
    plot_and_save_confusion_matrix(
        y_true=all_true_labels,
        y_pred=all_pred_labels,
        title="Final Confusion Matrix (All Folds)",
        filename="final_confusion_matrix.png",
        output_dir=output_dir
    )

    plot_custom_cv(extracted_features)


def train_and_evaluate(X_train, y_train, X_test, y_test, model_name, n_components):
    classifiers = get_classifiers()
    results = {}
    classifier_preds = {}  # To store predictions for late fusion
    
    # Manually create 3 folds as per your dataset structure
    folds = [
        (list(range(0, 300)), list(range(300, 600)), list(range(600, 900))),
        (list(range(300, 600)), list(range(600, 900)), list(range(0, 300))),
        (list(range(600, 900)), list(range(0, 300)), list(range(300, 600)))
    ]

    for fold_idx, (train_indices, val_indices, test_indices) in enumerate(folds):
        print(f"Training and tuning for Fold {fold_idx + 1}...")
        
        # Prepare the training, validation, and test sets based on fold indices
        X_train_fold, y_train_fold = X_train[train_indices], y_train[train_indices]
        X_val_fold, y_val_fold = X_train[val_indices], y_train[val_indices]
        X_test_fold, y_test_fold = X_test[test_indices], y_test[test_indices]
        
        # Train the classifiers with GridSearchCV (you can adjust this for each fold)
        for name, clf_dict in classifiers.items():
            print(f"Training and tuning {name}...")

            # Set up GridSearchCV with KFold
            grid = GridSearchCV(clf_dict['model'], clf_dict['params'], cv=3, scoring='accuracy', n_jobs=-1)
            grid.fit(X_train_fold, y_train_fold)
            
            best_model = grid.best_estimator_
            print(f"{name} - Best Parameters: {grid.best_params_}")

            # Collect predicted probabilities for fusion
            y_probs = best_model.predict_proba(X_test_fold)
            classifier_preds[name] = y_probs

            # Evaluate individual classifiers
            y_pred = best_model.predict(X_test_fold)   # np.argmax(y_probs, axis=1)
            accuracy = accuracy_score(y_test_fold, y_pred)
            report = classification_report(y_test_fold, y_pred, output_dict=True)

            print(f"Accuracy for fold {fold_idx + 1}, classifier {name}: {accuracy}")
            print("Classification Report:\n", report)

            # Optionally, collect training accuracy
            train_pred = best_model.predict(X_train_fold)
            train_accuracy = accuracy_score(y_train_fold, train_pred)
            print("***TRAINING ACCURACY")
            print(train_accuracy)

        # Validate

        # val_images = scale_test_features = scaler.transform(validation_features)



        # # Make predictions on the validation set
        # y_val_pred = best_model.predict(val_images)

        # # Evaluate the performance
        # val_accuracy = accuracy_score(val_labels, y_val_pred)
        # val_report = classification_report(val_labels, y_val_pred)

        # # Print results
        # print(f"Validation Accuracy: {val_accuracy:.4f}")
        # print("Validation Classification Report:")
        # print(val_report)

        print("Training Predictions:", np.unique(train_pred, return_counts=True))
        print("Test Predictions:", np.unique(y_pred, return_counts=True))
        print("Actual Test Labels:", np.unique(y_test, return_counts=True))

        results[name] = {
            "train_accuracy": f"{train_accuracy:.4f}",  # Store training accuracy
            "accuracy":  f"{accuracy:.4f}", # Test accuracy
            "report": report,
            "best_params": grid.best_params_,
            # "val_accuracy": val_accuracy,
            # "val_report": val_report
        }
        print(f"{name} - Training Accuracy: {train_accuracy:.4f}")  # Add this
        print(f"{name} - Test Accuracy: {accuracy}")
        print(f"{name} - Classification Report:\n", report)
        # Generate and save confusion matrix
        output_dir = os.path.join("results", model_name, f"pca_{n_components}")
        confusion_matrix_path = os.path.join(output_dir, f"{name}_confusion_matrix.png")
        plot_and_save_confusion_matrix(y_test, y_pred, f"Confusion Matrix - {name}", confusion_matrix_path, output_dir)

    # Late fusion strategies
    print("\nApplying late fusion strategies...")

    # Initialize fused predictions
    fused_probs_sum = np.zeros_like(list(classifier_preds.values())[0])  # Sum fusion
    fused_probs_product = np.zeros_like(list(classifier_preds.values())[0])  # Log-transformed Product fusion
    fused_probs_average = np.zeros_like(list(classifier_preds.values())[0])  # Average fusion
    fused_probs_min = np.ones_like(list(classifier_preds.values())[0]) * np.inf  # Min fusion
    fused_probs_max = np.zeros_like(list(classifier_preds.values())[0])  # Max fusion

    # Combine predictions
    for probs in classifier_preds.values():
        fused_probs_sum += probs  # Summing probabilities
        fused_probs_product += np.log(probs + 1e-10)  # Log-transformed product fusion
        fused_probs_average += probs  # Average fusion (will divide later)
        fused_probs_min = np.minimum(fused_probs_min, probs)  # Min fusion
        fused_probs_max = np.maximum(fused_probs_max, probs)  # Max fusion

    # Final transformations
    fused_probs_product = np.exp(fused_probs_product)  # Convert back to normal scale
    fused_probs_average /= len(classifier_preds)  # Average fusion

    # Normalize min fusion
    fused_probs_min = (fused_probs_min - np.min(fused_probs_min, axis=1, keepdims=True)) / (
        np.max(fused_probs_min, axis=1, keepdims=True) - np.min(fused_probs_min, axis=1, keepdims=True) + 1e-10
    )

    # Convert probabilities to class labels
    y_pred_sum = np.argmax(fused_probs_sum, axis=1)
    y_pred_product = np.argmax(fused_probs_product, axis=1)
    y_pred_average = np.argmax(fused_probs_average, axis=1)
    y_pred_min = np.argmax(fused_probs_min, axis=1)
    y_pred_max = np.argmax(fused_probs_max, axis=1)

    # Evaluate fusion results
    fusion_results = {
        "Sum Fusion": {
            "accuracy": accuracy_score(y_test, y_pred_sum),
            "report": classification_report(y_test, y_pred_sum, output_dict=True)
        },
        "Product Fusion": {
            "accuracy": accuracy_score(y_test, y_pred_product),
            "report": classification_report(y_test, y_pred_product, output_dict=True)
        },
        "Average Fusion": {
            "accuracy": accuracy_score(y_test, y_pred_average),
            "report": classification_report(y_test, y_pred_average, output_dict=True)
        },
        "Min Fusion": {
            "accuracy": accuracy_score(y_test, y_pred_min),
            "report": classification_report(y_test, y_pred_min, output_dict=True)
        },
        "Max Fusion": {
            "accuracy": accuracy_score(y_test, y_pred_max),
            "report": classification_report(y_test, y_pred_max, output_dict=True)
        }
    }

    print("\nFusion Results:")
    for fusion_name, fusion_metrics in fusion_results.items():
        print(f"{fusion_name} - Accuracy: {fusion_metrics['accuracy']}")
        print(f"{fusion_name} - Classification Report:\n", fusion_metrics['report'])

    # Add fusion results to the overall results
    results["Fusion"] = fusion_results

    return results



