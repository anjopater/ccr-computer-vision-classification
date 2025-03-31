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
from utils.save_plots import plot_and_save_confusion_matrix, plot_cv_indices

from sklearn.preprocessing import StandardScaler
import os

def get_classifiers():
    return {
        # 'KNN': {
        #     'model': KNeighborsClassifier(),
        #     'params': {
        #         'n_neighbors': [1, 3, 5, 7, 9],
        #         'weights': ['uniform', 'distance'],
        #         'p': [1, 2]
        #     }
        # },
        # 'SVM': {
        #     'model': SVC(probability=True),
        #     'params': {
        #         'C': [0.1, 1, 10],
        #         'kernel': ['linear', 'rbf', "sigmoid", "poly"],
        #         'gamma': ['scale', 'auto']
        #     }
        # },
        # 'Random Forest': {
        #     'model': RandomForestClassifier(),
        #     'params': {
        #         'n_estimators': [50, 100, 200],
        #         'max_depth': [None, 10, 20],
        #         'min_samples_split': [2, 5],
        #         'min_samples_leaf': [1, 2]
        #     }
        # },
        'Logistic Regression': {
            'model': LogisticRegression(max_iter=500),
            'params': {
                'C': [0.1, 1, 10],
                'penalty': ['l2', 'none'],
                'solver': ['lbfgs', 'saga']
            }
        },
        # 'MLP': {
        #     'model': MLPClassifier(random_state=42),
        #     'params': {
        #         'hidden_layer_sizes': [(128, 64, 32)],
        #         'activation': ['relu', 'logistic', 'tanh', 'identity'],
        #         'solver': ['adam', 'sgd', 'lbfgs'],
        #         'learning_rate': ['constant', 'adaptive', 'invscaling'],
        #         'max_iter': [200, 500, 1000],
        #         'alpha': [0.0001, 0.001, 0.01]  # Regularization strength
        #     }
        # },    
        # 'MLP2': {
        #     'model': MLPClassifier(random_state=42),
        #     'params': {
        #         'hidden_layer_sizes': [(64,), (64, 32)],
        #         'activation': ['relu', 'logistic', 'tanh', 'identity'],
        #         'solver': ['adam', 'sgd', 'lbfgs'],
        #         'learning_rate': ['constant', 'adaptive', 'invscaling'],
        #         'max_iter': [200, 500, 1000],
        #         'alpha': [0.0001, 0.001, 0.01]  # Regularization strength
        #     }
        # },
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

def train_and_evaluate(X_train, y_train, X_test, y_test, train_groups, test_groups, model_name, n_components):
    classifiers = get_classifiers()
    results = {}
    classifier_preds = {}  # To store predictions for late fusion

    for name, clf_dict in classifiers.items():
        print(f"Training and tuning {name}...")
        cv = GroupKFold(n_splits=4)

        # Plot CV indices
        output_dir = os.path.join("results", model_name, f"pca_{n_components}")
        cv_plot_path = plot_cv_indices(cv, X_train, y_train, train_groups, n_splits=4, output_dir=output_dir)
        print(f"CV indices plot saved to: {cv_plot_path}")

        grid = GridSearchCV(clf_dict['model'], clf_dict['params'], cv=cv, scoring='accuracy', n_jobs=-1)
        print(y_test)
        print(test_groups)
        print("Groups in Training:", train_groups)
        print("Groups in Testing:", test_groups)
        grid.fit(X_train, y_train, groups=train_groups)
        best_model = grid.best_estimator_
        print(f"{name} - Best Parameters: {grid.best_params_}")

        # Collect predicted probabilities for fusion
        y_probs = best_model.predict_proba(X_test)
        classifier_preds[name] = y_probs

        # Evaluate individual classifiers
        y_pred = best_model.predict(X_test)   #np.argmax(y_probs, axis=1)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)

        # 
        train_pred = best_model.predict(X_train)
        train_accuracy = accuracy_score(y_train, train_pred)
        print("***TRAINING")
        print(train_pred)
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

    # # Late fusion strategies
    # print("\nApplying late fusion strategies...")

    # # Initialize fused predictions
    # fused_probs_sum = np.zeros_like(list(classifier_preds.values())[0])  # Sum fusion
    # fused_probs_product = np.zeros_like(list(classifier_preds.values())[0])  # Log-transformed Product fusion
    # fused_probs_average = np.zeros_like(list(classifier_preds.values())[0])  # Average fusion
    # fused_probs_min = np.ones_like(list(classifier_preds.values())[0]) * np.inf  # Min fusion
    # fused_probs_max = np.zeros_like(list(classifier_preds.values())[0])  # Max fusion

    # # Combine predictions
    # for probs in classifier_preds.values():
    #     fused_probs_sum += probs  # Summing probabilities
    #     fused_probs_product += np.log(probs + 1e-10)  # Log-transformed product fusion
    #     fused_probs_average += probs  # Average fusion (will divide later)
    #     fused_probs_min = np.minimum(fused_probs_min, probs)  # Min fusion
    #     fused_probs_max = np.maximum(fused_probs_max, probs)  # Max fusion

    # # Final transformations
    # fused_probs_product = np.exp(fused_probs_product)  # Convert back to normal scale
    # fused_probs_average /= len(classifier_preds)  # Average fusion

    # # Normalize min fusion
    # fused_probs_min = (fused_probs_min - np.min(fused_probs_min, axis=1, keepdims=True)) / (
    #     np.max(fused_probs_min, axis=1, keepdims=True) - np.min(fused_probs_min, axis=1, keepdims=True) + 1e-10
    # )

    # # Convert probabilities to class labels
    # y_pred_sum = np.argmax(fused_probs_sum, axis=1)
    # y_pred_product = np.argmax(fused_probs_product, axis=1)
    # y_pred_average = np.argmax(fused_probs_average, axis=1)
    # y_pred_min = np.argmax(fused_probs_min, axis=1)
    # y_pred_max = np.argmax(fused_probs_max, axis=1)

    # # Evaluate fusion results
    # fusion_results = {
    #     "Sum Fusion": {
    #         "accuracy": accuracy_score(y_test, y_pred_sum),
    #         "report": classification_report(y_test, y_pred_sum, output_dict=True)
    #     },
    #     "Product Fusion": {
    #         "accuracy": accuracy_score(y_test, y_pred_product),
    #         "report": classification_report(y_test, y_pred_product, output_dict=True)
    #     },
    #     "Average Fusion": {
    #         "accuracy": accuracy_score(y_test, y_pred_average),
    #         "report": classification_report(y_test, y_pred_average, output_dict=True)
    #     },
    #     "Min Fusion": {
    #         "accuracy": accuracy_score(y_test, y_pred_min),
    #         "report": classification_report(y_test, y_pred_min, output_dict=True)
    #     },
    #     "Max Fusion": {
    #         "accuracy": accuracy_score(y_test, y_pred_max),
    #         "report": classification_report(y_test, y_pred_max, output_dict=True)
    #     }
    # }

    # print("\nFusion Results:")
    # for fusion_name, fusion_metrics in fusion_results.items():
    #     print(f"{fusion_name} - Accuracy: {fusion_metrics['accuracy']}")
    #     print(f"{fusion_name} - Classification Report:\n", fusion_metrics['report'])

    # # Add fusion results to the overall results
    # results["Fusion"] = fusion_results

    return results