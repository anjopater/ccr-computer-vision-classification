# utils/evaluator.py
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import numpy as np
from sklearn.naive_bayes import GaussianNB
import os

def get_classifiers(random_state=42):
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
        #     'model': SVC(probability=True, random_state=random_state),
        #     'params': {
        #         'C': [0.1, 1, 10],
        #         'kernel': ['linear', 'rbf'],
        #         'gamma': ['scale']
        #     }
        # },
        # 'Random Forest': {
        #     'model': RandomForestClassifier(random_state=random_state),
        #     'params': {
        #         'n_estimators': [50, 100, 200],
        #         'max_depth': [None, 10, 20],
        #         'min_samples_split': [2, 5],
        #         'min_samples_leaf': [1, 2]
        #     }
        # },
            'Logistic Regression': {
                'model': LogisticRegression(max_iter=500, random_state=random_state),
                'params': {
                    'C': [0.1, 1, 10],
                    'penalty': ['l2', None],
                    'solver': ['lbfgs', 'saga'],
                    'max_iter': [500, 1000, 2000, 5000],

                }
            },
        # 'MLP': {
        #     'model': MLPClassifier(random_state=42, shuffle=False),
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
        #     'model': MLPClassifier(random_state=42, shuffle=False),
        #     'params': {
        #         'hidden_layer_sizes': [(64,), (64, 32)],
        #         'activation': ['relu', 'logistic', 'tanh', 'identity'],
        #         'solver': ['adam', 'sgd', 'lbfgs'],
        #         'learning_rate': ['constant', 'adaptive', 'invscaling'],
        #         'max_iter': [200, 500, 1000],
        #         'alpha': [0.0001, 0.001, 0.01]  # Regularization strength
        #     }
        # },
        # 'GaussianNB': {
        #     'model': GaussianNB(),
        #     'params': {
        #         'var_smoothing': [1e-9, 1e-8, 1e-7]
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