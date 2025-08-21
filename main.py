# main.py
# ───────────────────────────────────────────────────────────────
"""
Nested CV over PCA components and multiple classifiers
for each CNN feature extractor.  Adds:
  • weighted soft-voting ensemble
"""

import os, warnings, numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from sklearn.ensemble   import VotingClassifier, StackingClassifier
from sklearn.metrics  import accuracy_score, classification_report
from sklearn.exceptions import ConvergenceWarning
from sklearn.base import BaseEstimator
from sklearn.calibration import CalibratedClassifierCV

from config                  import PCA_COMPONENTS, MODELS, RESULTS_FILE
from utils.data_loader       import load_data, load_or_extract
from utils.feature_extractor import extract_cnn_features, extract_handcrafted_features
from utils.evaluator         import get_classifiers
from utils.logger            import save_results
from utils.save_plots        import plot_and_save_confusion_matrix, plot_fold_animal_heatmap, plot_multiple_roc, plot_and_save_umap, compute_tsne
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier
from umap import UMAP


from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer
)

warnings.filterwarnings("ignore", category=ConvergenceWarning)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# list of scalers to sweep
SCALERS = [
    StandardScaler(),                          # z-score (baseline)
    MinMaxScaler(),                            # [0, 1] range
    RobustScaler(),                            # median / IQR
    PowerTransformer(method="yeo-johnson")     # de-skew + z-score
]
CACHE_DIR = "cached_features"
os.makedirs(CACHE_DIR, exist_ok=True)

def clean_params(d):
    """Troca objetos sklearn por seus nomes de classe."""
    return {
        k: (v.__class__.__name__           # → "PowerTransformer"
            if isinstance(v, BaseEstimator)
            else v)
        for k, v in d.items()
    }

def main():
    # 1) Load data (hold-out split already group-aware)
    train_imgs, train_lbl, train_grp, \
    test_imgs,  test_lbl,  test_grp = load_data()

    results = {}
    # 2) One pass per feature extractor
    for model_name, model_key in MODELS.items():
        score_dict = {}
        print(f"\n=== Processing extractor: {model_name} ===")

        # 2a) Extract features
        if model_key == "handcrafted":            
            train_cache = os.path.join(CACHE_DIR, "cache_train.joblib")
            test_cache  = os.path.join(CACHE_DIR, "cache_test.joblib")

            X_train = load_or_extract(train_imgs, train_cache,
                                    extract_handcrafted_features)
            X_test  = load_or_extract(test_imgs,  test_cache,
                                    extract_handcrafted_features)
        else: # CNN-based extractor
            X_train = extract_cnn_features(train_imgs, model_key)
            X_test  = extract_cnn_features(test_imgs,  model_key)

        classifiers = get_classifiers()
        best_pipelines = {}
        results[model_name] = {}

        # 2b) Inner loop: tune each classifier
        for clf_name, clf_dict in classifiers.items():
            print(f"--- {clf_name} ---")

            # Decide whether to include PCA
            if model_key == "handcrafted":
                # low-dim handcrafted features → skip PCA
                pipe = Pipeline([
                    ('scaler', StandardScaler()),
                    ('clf',    clf_dict['model'])
                ])
                param_grid = {
                    'scaler': SCALERS,                     # ← vary the scaler
                    **{f"clf__{p}": vals                   # keep the classifier’s own params
                    for p, vals in clf_dict['params'].items()}
                }
                
            else:
                # high-dim CNN embeddings → use PCA
                pipe = Pipeline([
                    ('scaler', StandardScaler()),
                    ('pca',    PCA(random_state=42)),
                    ('clf',    clf_dict['model'])
                ])
                param_grid = {'pca__n_components': PCA_COMPONENTS, 'scaler': SCALERS}
                param_grid.update({
                    f"clf__{p}": vals
                    for p, vals in clf_dict['params'].items()
                })

            cv_inner = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=42)
            
            # --- DEBUG: log label balance per fold ---
            print("Inner CV class balance (train vs. val in each fold):")
            for fold, (tr_idx, val_idx) in enumerate(
                    cv_inner.split(X_train, train_lbl, train_grp), start=1):
                tr_cnt = np.bincount(train_lbl[tr_idx], minlength=2)
                vl_cnt = np.bincount(train_lbl[val_idx],  minlength=2)
                print(f" Fold {fold}: "
                    f" train → C:{tr_cnt[0]} CRC:{tr_cnt[1]};"
                    f" val → C:{vl_cnt[0]} CRC:{vl_cnt[1]}")
            # ---------------------------------------------
            
            grid = GridSearchCV(
                pipe, param_grid,
                cv=cv_inner, scoring='f1',
                n_jobs=-1, refit=True
            )
            grid.fit(X_train, train_lbl, groups=train_grp)
            
            print(f"--- Scores per fold for best {clf_name} ---")
            best_model_index = grid.best_index_
            fold_scores = []
            for i in range(cv_inner.get_n_splits()):
                fold_score_key = f"split{i}_test_score"
                score = grid.cv_results_[fold_score_key][best_model_index]
                fold_scores.append(score)
                print(f"  Fold {i+1}: {score:.4f}")
            
            print(f"List of scores for {clf_name}: {np.round(fold_scores, 4).tolist()}")

            # # 2. Perform hold-out evaluation on the final test set
            # best_model = grid.best_estimator_
            # y_pred = best_model.predict(X_test)
            # y_score = best_model.predict_proba(X_test)[:, 1]
            # score_dict[clf_name] = y_score
            # acc = accuracy_score(test_lbl, y_pred)
            # best_pipelines[clf_name] = best_model

            # # 3. Save ALL results for this classifier in ONE place
            # results[model_name][clf_name] = {
                
            #     "fold_scores": np.round(fold_scores, 4).tolist() # <-- ADDED a key for fold_scores
            # }
            
            # choose an output folder
            heatmap_dir = os.path.join("results", model_name, clf_name, "cv_folds")
            os.makedirs(heatmap_dir, exist_ok=True)
            # plot the heatmap of train vs val animals
            plot_fold_animal_heatmap(
                cv_inner,
                train_grp,
                train_lbl,
                heatmap_dir
            )

            # Hold-out evaluation
            y_pred = grid.best_estimator_.predict(X_test)
            y_score = grid.best_estimator_.predict_proba(X_test)[:, 1]   # ← probability for class 1
            score_dict[clf_name] = y_score

            acc  = accuracy_score(test_lbl, y_pred)

            best_pipelines[clf_name] = grid.best_estimator_

            # Save confusion matrix for classifier
            out_dir = os.path.join("results", model_name, clf_name)
            os.makedirs(out_dir, exist_ok=True)
            plot_and_save_confusion_matrix(
                test_lbl, y_pred,
                title=f"Confusion Matrix – {clf_name}",
                filename="confmat.png",
                output_dir=out_dir
            )
            
            # add results in a json structure
            results[model_name][clf_name] = {
                "test_accuracy"        : f"{acc:.4f}",
                "best_params"          : clean_params(grid.best_params_),
                "classification_report": classification_report(
                                              test_lbl, y_pred, output_dict=True),
                "fold_scores": np.round(fold_scores, 4).tolist()
            }
            
            # plot data distributions
            X_all   = np.vstack([X_train, X_test])
            y_all   = np.hstack([train_lbl, test_lbl])
            scaler = PowerTransformer(method='yeo-johnson')    
            X_scaled  = scaler.fit_transform(X_all)

            # UMAP
            X_emb   = UMAP(random_state=42).fit_transform(X_scaled )
            plot_and_save_umap(X_emb, y_all, 'UMAP', out_dir+ '/umap.png')
            
           # t-SNE
            X_tsne = compute_tsne(
                X_scaled,
                n_components=2,
                perplexity=30,
                random_state=42
            )

            plot_and_save_umap(                
                X_tsne, y_all,
                title='t-SNE',
                filename= out_dir + '/tsne.png'
            )
                        
        # ROC curve for al classifiers
        # in the same place you already calculate disagree/corr matrices
        all_roc_dir = os.path.join("results", model_name)
        plot_multiple_roc(
            test_lbl,
            score_dict,
            title=f"All ROC – {model_name}",
            filename="roc_all_classifiers.png",
            output_dir=all_roc_dir
        )
        
        print("Models collected:", list(best_pipelines.keys()))

        # # 2d) Ensembles (only if we have any base learners)
        # if not best_pipelines:
        #     print(f"No base learners for {model_name}, skipping ensembles.")
        # else:
        #     chosen  = ['SVM', 'Logistic Regression', 'Random Forest', 'GaussianNB']
        #     weights = [12, 10, 8, 6, 4]

        #     # ---- Calibrators (SVM precisa; RF opcional) ----
        #     svm_cal = CalibratedClassifierCV(best_pipelines['SVM'],  cv=3, method='sigmoid')
        #     rf_cal  = CalibratedClassifierCV(best_pipelines['Random Forest'], cv=3, method='sigmoid')

        #     estimators = [
        #         ('svm', svm_cal),
        #         ('lr',  best_pipelines['Logistic Regression']),
        #         ('rf',  rf_cal),
        #         ('mlp2',best_pipelines['MLP2']),
        #         ('nb',  best_pipelines['GaussianNB'])
        #     ]

        #     soft_vote = VotingClassifier(
        #         estimators=estimators,
        #         voting='soft',
        #         weights=weights,
        #         n_jobs=-1
        #     )
        #     soft_vote.fit(X_train, train_lbl)
        #     y_soft = soft_vote.predict(X_test)

        #     results[model_name]['SoftVoting'] = {
        #         "members"      : chosen,
        #         "weights"      : dict(zip(chosen, weights)),
        #         "test_accuracy": f"{accuracy_score(test_lbl, y_soft):.4f}",
        #         "report"       : classification_report(test_lbl, y_soft, output_dict=True)
        #     }
            
        #     # Stacking
        #     lr      = best_pipelines['Logistic Regression']
        #     nb      = best_pipelines['GaussianNB']

        #     # 2) estimator fixed list
        #     estimators = [
        #         ('svm', svm_cal),
        #         ('lr',  lr),
        #         ('rf',  rf_cal),
        #         ('nb',  nb)
        #     ]

        #     # 3) Defining Meta classifier
        #     meta_clf = GradientBoostingClassifier(random_state=42)

        #     # 4) Creating the meta classifer
        #     stack = StackingClassifier(
        #         estimators=estimators,
        #         final_estimator=meta_clf,
        #         cv=4,                   
        #         n_jobs=-1,
        #         passthrough=False        # pass only the probabilities
        #     )

        #     # 5) Treine e avalie
        #     stack.fit(X_train, train_lbl)
        #     y_pred = stack.predict(X_test)
        #     acc    = accuracy_score(test_lbl, y_pred)
        #     print(f"Stacking accuracy: {acc:.4f}")
                        
        #     # Save confusion matrix for soft voting
        #     out_dir = os.path.join("results", model_name)
        #     os.makedirs(out_dir, exist_ok=True)
        #     plot_and_save_confusion_matrix(
        #         test_lbl, y_soft,
        #         title=f"Confusion Matrix – {clf_name}",
        #         filename="confmat_softvoting.png",
        #         output_dir=out_dir
        #     )

        #     results[model_name]['Stacking'] = {
        #         "base_learners": 'svm,rf,lr,nb',
        #         "meta": "LogisticRegression",
        #         "test_accuracy": f"{accuracy_score(test_lbl, y_pred):.4f}",
        #         "report": classification_report(test_lbl, y_pred, output_dict=True)
        #     }
            

    # 3) Persist all results
    save_results(results, RESULTS_FILE)


if __name__ == "__main__":
    main()
