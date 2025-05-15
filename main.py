# main.py
"""
Nested cross-validation over PCA components and multiple classifiers
for each CNN feature extractor. Ensures a true hold-out test set.
"""
import os
import warnings
import ssl
import numpy as np

from config import (
    C_PATH, CCR_PATH,
    PCA_COMPONENTS, MODELS,
    RESULTS_FILE
)
from utils.data_loader import load_data
from utils.feature_extractor import extract_cnn_features
from utils.evaluator import get_classifiers
from utils.logger import save_results
from utils.save_plots import (
    plot_and_save_explained_variance,
    plot_and_save_correlation_heatmap,
    generate_plots_and_save_results,
    plot_and_save_confusion_matrix,
    plot_cv_indices,
    plot_fold_animal_heatmap
)
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.ensemble import VotingClassifier
import pandas as pd
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# ----------------------------------------------------------------------------
# 1) Suppress warnings to keep the output clean
# ----------------------------------------------------------------------------
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ----------------------------------------------------------------------------
# 2) Main execution: load data, nested CV, final test eval, save results
# ----------------------------------------------------------------------------
def main():
    # 2a) Load and split the data (group-wise to avoid animal overlap)
    train_imgs, train_labels, train_groups, \
    test_imgs, test_labels, test_groups = load_data()

    # Container for all results
    results = {}

    # 2b) Iterate over each CNN feature extractor specified in config
    for model_name, model_key in MODELS.items():
        print(f"\n=== Processing CNN: {model_name} ===")

        # Extract features for train/test once per CNN
        X_train = extract_cnn_features(train_imgs, model_key)
        X_test  = extract_cnn_features(test_imgs, model_key)

        # Retrieve your classifier dict: name -> {'model':..., 'params':...}
        classifiers = get_classifiers()

        # Prepare sub-dict for this CNN
        results[model_name] = {}
        # A dict to hold the best‐fitted pipelines for each classifier
        best_pipelines = {}

        # 2c) Loop over each classifier
        for clf_name, clf_dict in classifiers.items():
            print(f"--- Classifier: {clf_name} on {model_name} features ---")

            # Build sklearn Pipeline: scale -> PCA -> classifier
            pipe = Pipeline([
                ('scaler', StandardScaler()),
                ('pca',     PCA(random_state=42)),
                ('clf',     clf_dict['model']),
            ])

            # Build hyperparameter grid:
            # - pca__n_components from config
            # - classifier params, prefixed with 'clf__'
            param_grid = {'pca__n_components': PCA_COMPONENTS}
            param_grid.update({
                f"clf__{param}": values
                for param, values in clf_dict['params'].items()
            })

            # Nested CV: tune on training set only, use GroupKFold to respect groups
            cv_inner = GroupKFold(n_splits=4)
            grid = GridSearchCV(
                estimator=pipe,
                param_grid=param_grid,
                cv=cv_inner,
                scoring='accuracy',
                n_jobs=-1,
                refit=True
            )
            
     
            
            # Fit (tunes PCA + classifier hyperparams)
            grid.fit(X_train, train_labels, groups=train_groups)
            best_params = grid.best_params_
            print(f"Best params: {best_params}")

            # Plot PCA diagnostics for chosen number of components
            best_pca = grid.best_estimator_.named_steps['pca']
            # plot_and_save_explained_variance(best_pca, f"{model_name}_{clf_name}", best_params['pca__n_components'])
            # plot_and_save_correlation_heatmap(best_pca, f"{model_name}_{clf_name}", best_params['pca__n_components'])

            # Evaluate the tuned pipeline on the untouched hold-out test set
            y_pred = grid.best_estimator_.predict(X_test)
            acc = accuracy_score(test_labels, y_pred)
            report_dict = classification_report(test_labels, y_pred, output_dict=True)
            best_pipelines[clf_name] = grid.best_estimator_
            print("Pipelines collected:", list(best_pipelines.keys()))

            print(f"Test Accuracy for {clf_name}: {acc:.4f}")
            print(classification_report(test_labels, y_pred))
            
                   # ======================================================================
            # Plot & log how animals are split in each inner CV fold
            # ======================================================================
            # choose an output folder under this classifier’s PCA folder
            fold_vis_dir = os.path.join(
                "results", model_name, clf_name, f"pca_{best_params['pca__n_components']}", "cv_folds"
            )
            # Note: if best_params isn’t available yet, just use a temp folder:
            fold_vis_dir = os.path.join("results", model_name, clf_name, "cv_folds")
            plot_cv_indices(
                cv_inner,
                X_train,
                train_labels,
                train_groups,
                n_splits=4,
                output_dir=fold_vis_dir
            )
            
            heatmap_dir = os.path.join("results", model_name, "cv_folds")
            plot_fold_animal_heatmap(cv_inner, train_groups, n_splits=4, output_dir=heatmap_dir)

            # Save to results dict
        # Append this trial’s results to a list under this classifier name
            results[model_name].setdefault(clf_name, []).append({
                # If you want to see which PCA size was chosen:
                'n_components':        best_params['pca__n_components'],
                'best_params':         best_params,
                'test_accuracy':       f"{acc:.4f}",
                'classification_report': report_dict
            })

            # # Optional: generate any feature histograms / result plots you have
            # generate_plots_and_save_results(
            #     X_train, None, train_labels,
            #     None,   test_labels,   best_pca,
            #     os.path.join('results', model_name, clf_name)
            # )
            
            # --- ENSEMBLE SECTION: soft-voting over all your tuned classifiers ---
    # ----------------------------------------------------------------------------
            # 3) Measure pairwise disagreement & probability correlation among classifiers
            # ----------------------------------------------------------------------------
            print("\n--- Measuring diversity among classifiers ---")
            # Collect predictions
            pred_df = pd.DataFrame({
                name: pipe.predict(X_test)
                for name, pipe in best_pipelines.items()
            })
            # Disagreement rates
            names = pred_df.columns.tolist()
            disagree = pd.DataFrame(0.0, index=names, columns=names)
            for i, a in enumerate(names):
                for j, b in enumerate(names):
                    if i < j:
                        d = (pred_df[a] != pred_df[b]).mean()
                        disagree.loc[a, b] = disagree.loc[b, a] = d
            print("Pairwise disagreement rates:")
            print(disagree)

            # Probability correlations
            prob_df = pd.DataFrame({
                name: pipe.predict_proba(X_test)[:,1]
                for name, pipe in best_pipelines.items()
            })
            corr = prob_df.corr()
            print("Pairwise probability correlation:")
            print(corr)

            # Save diversity metrics
            results[model_name]['disagreement_matrix'] = disagree.to_dict()
            results[model_name]['correlation_matrix'] = corr.to_dict()

            # # ----------------------------------------------------------------------------
            # # 4) Build a soft-voting ensemble of all classifiers
            # # ----------------------------------------------------------------------------
            # print(f"--- Building ensemble over: {list(best_pipelines.keys())} ---")
            # ensemble = VotingClassifier(
            #     estimators=[(n, p) for n, p in best_pipelines.items()],
            #     voting='soft',
            #     n_jobs=-1
            # )
            # # Retrain on full training set
            # ensemble.fit(X_train, train_labels)

            # # Evaluate ensemble
            # y_ens = ensemble.predict(X_test)
            # ens_acc = accuracy_score(test_labels, y_ens)
            # ens_report = classification_report(test_labels, y_ens, output_dict=True)
            # print(f"Ensemble Test Accuracy: {ens_acc:.4f}")
            # print(classification_report(test_labels, y_ens))

            # # Save ensemble results
            # results[model_name]['Ensemble'] = {
            #     'test_accuracy':        f"{ens_acc:.4f}",
            #     'classification_report': ens_report,
            #     'members':              list(best_pipelines.keys())
            # }
            
            # === NEW: make & save confusion matrix ===
            # build output directory path
            out_dir = os.path.join("results", model_name, clf_name, f"pca_{best_params['pca__n_components']}")
            os.makedirs(out_dir, exist_ok=True)

            plot_and_save_confusion_matrix(
                y_true=test_labels,
                y_pred=y_pred,
                title=f"Confusion Matrix – {clf_name}",
                filename=f"{clf_name}_confusion_matrix.png",
                output_dir=out_dir
            )
            # ============================================

    # 2d) Persist all results to JSON
    save_results(results, RESULTS_FILE)


if __name__ == '__main__':
    main()
