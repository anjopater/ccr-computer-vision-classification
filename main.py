# main.py
# ───────────────────────────────────────────────────────────────
"""
Nested CV over PCA components and multiple classifiers
for each CNN feature extractor.  Adds:
  • weighted soft-voting ensemble
  • stacking meta-learner
"""

import os, warnings, ssl, numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV, GroupShuffleSplit
from sklearn.ensemble   import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics  import accuracy_score, classification_report
from sklearn.exceptions import ConvergenceWarning

from config                  import PCA_COMPONENTS, MODELS, RESULTS_FILE
from utils.data_loader       import load_data
from utils.feature_extractor import extract_cnn_features, extract_haralick_granulo, feature_extractor
from utils.evaluator         import get_classifiers
from utils.logger            import save_results
from utils.save_plots        import plot_and_save_confusion_matrix, plot_fold_animal_heatmap, plot_multiple_roc, plot_and_save_roc

warnings.filterwarnings("ignore", category=ConvergenceWarning)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def main():
    # 1) Load data (hold-out split already group-aware)
    train_imgs, train_lbl, train_grp, \
    test_imgs,  test_lbl,  test_grp = load_data()

    results = {}
    score_dict = {}
    # 2) One pass per feature extractor
    for model_name, model_key in MODELS.items():
        print(f"\n=== Processing extractor: {model_name} ===")

        # 2a) Extract features
        if model_key == "haralick_granulo":
            X_train = extract_haralick_granulo(train_imgs)
            X_test  = extract_haralick_granulo(test_imgs)
        else:
            X_train = extract_cnn_features(train_imgs, model_key)
            X_test  = extract_cnn_features(test_imgs,  model_key)

        classifiers    = get_classifiers()
        best_pipelines = {}
        results[model_name] = {}

        # 2b) Inner loop: tune each classifier
        for clf_name, clf_dict in classifiers.items():
            print(f"--- {clf_name} ---")

            # Decide whether to include PCA
            if model_key == "haralick_granulo":
                # low-dim handcrafted features → skip PCA
                pipe = Pipeline([
                    ('scaler', StandardScaler()),
                    ('clf',    clf_dict['model'])
                ])
                param_grid = {
                    f"clf__{p}": vals
                    for p, vals in clf_dict['params'].items()
                }
            else:
                # high-dim CNN embeddings → use PCA
                pipe = Pipeline([
                    ('scaler', StandardScaler()),
                    ('pca',    PCA(random_state=42)),
                    ('clf',    clf_dict['model'])
                ])
                param_grid = {'pca__n_components': PCA_COMPONENTS}
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
                cv=cv_inner, scoring='accuracy',
                n_jobs=-1, refit=True
            )
            grid.fit(X_train, train_lbl, groups=train_grp)
            
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

            acc    = accuracy_score(test_lbl, y_pred)


     

            best_pipelines[clf_name] = grid.best_estimator_

            # Save confusion matrix
            out_dir = os.path.join("results", model_name, clf_name)
            os.makedirs(out_dir, exist_ok=True)
            plot_and_save_confusion_matrix(
                test_lbl, y_pred,
                title=f"Confusion Matrix – {clf_name}",
                filename="confmat.png",
                output_dir=out_dir
            )
            
                   # ROC curve
            # in the same place you already calculate disagree/corr matrices
            all_roc_dir = os.path.join("results", model_name)
            plot_multiple_roc(
                test_lbl,
                score_dict,
                title=f"All ROC – {model_name}",
                filename="roc_all_classifiers.png",
                output_dir=all_roc_dir
            )

            results[model_name][clf_name] = {
                "test_accuracy"        : f"{acc:.4f}",
                "best_params"          : grid.best_params_,
                "classification_report": classification_report(
                                              test_lbl, y_pred, output_dict=True),
            }
            

        print("Models collected:", list(best_pipelines.keys()))

        # 2c) Diversity metrics
        pred_df = pd.DataFrame({
            n: m.predict(X_test)
            for n, m in best_pipelines.items()
        })
        names = pred_df.columns.tolist()
        disagree = pd.DataFrame(0.0, index=names, columns=names)
        for i, a in enumerate(names):
            for j, b in enumerate(names):
                if i < j:
                    v = (pred_df[a] != pred_df[b]).mean()
                    disagree.loc[a, b] = disagree.loc[b, a] = v

        prob_df = pd.DataFrame({
            n: m.predict_proba(X_test)[:, 1]
            for n, m in best_pipelines.items()
        })
        corr = prob_df.corr()

        results[model_name]['disagreement_matrix'] = disagree.to_dict()
        results[model_name]['correlation_matrix']  = corr.to_dict()

        # 2d) Ensembles (only if we have any base learners)
        if not best_pipelines:
            print(f"No base learners for {model_name}, skipping ensembles.")
        else:
            # Soft-voting (if required learners exist)
            if {'MLP2','KNN','Random Forest', 'GaussianNB'}.issubset(best_pipelines):
                soft_vote = VotingClassifier(
                    estimators=[
                        ('mlp2', best_pipelines['MLP2']),
                        ('nb',  best_pipelines['GaussianNB']),
                        ('rf',   best_pipelines['Random Forest'])
                    ],
                    voting='soft',
                    weights=[3,1,1],  # MLP2 has 60% influence
                    n_jobs=-1
                )
                soft_vote.fit(X_train, train_lbl)
                y_soft   = soft_vote.predict(X_test)
                soft_acc = accuracy_score(test_lbl, y_soft)
                soft_rep = classification_report(test_lbl, y_soft, output_dict=True)

                print(f"*** Soft-voting acc ({model_name}): {soft_acc:.4f}")

                results[model_name]['SoftVoting'] = {
                    "test_accuracy"        : f"{soft_acc:.4f}",
                    "classification_report": soft_rep,
                    "weights"              : {'mlp2':3, 'nb':1, 'rf':1}
                }

            # Stacking meta-learner
            base_names = list(best_pipelines.keys())
            n_train    = len(X_train)

            # (i) out-of-fold probability matrix
            oof_prob = np.zeros((n_train, len(base_names)))
            cv_outer = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=42)
            for tr_idx, val_idx in cv_outer.split(X_train, train_lbl, train_grp):
                for col, name in enumerate(base_names):
                    mdl = best_pipelines[name]
                    mdl.fit(X_train[tr_idx], train_lbl[tr_idx])
                    oof_prob[val_idx, col] = mdl.predict_proba(X_train[val_idx])[:, 1]

            # (ii) test-time probabilities
            test_prob = np.column_stack([
                best_pipelines[name].predict_proba(X_test)[:, 1]
                for name in base_names
            ])

            # (iii) train meta-learner
            meta = LogisticRegression(max_iter=2000)
            meta.fit(oof_prob, train_lbl)
            y_stack   = meta.predict(test_prob)
            stack_acc = accuracy_score(test_lbl, y_stack)
            stack_rep = classification_report(test_lbl, y_stack, output_dict=True)

            print(f"*** Stacking acc ({model_name}): {stack_acc:.4f}")

            results[model_name]['Stacking'] = {
                "test_accuracy"        : f"{stack_acc:.4f}",
                "classification_report": stack_rep,
                "base_learners"        : base_names,
                "meta_learner"         : "LogisticRegression"
            }

    # 3) Persist all results
    save_results(results, RESULTS_FILE)


if __name__ == "__main__":
    main()
