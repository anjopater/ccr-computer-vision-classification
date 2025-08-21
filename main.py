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
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


from config                  import PCA_COMPONENTS, MODELS, RESULTS_FILE
from utils.data_loader       import load_data, load_or_extract
from utils.feature_extractor import extract_cnn_features, extract_handcrafted_features
from utils.evaluator         import get_classifiers
from utils.logger            import save_results
from utils.save_plots        import plot_and_save_confusion_matrix, plot_fold_animal_heatmap, plot_multiple_roc, plot_and_save_umap, compute_tsne
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier
from umap import UMAP
import json
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression


from sklearn.cluster import KMeans
from imblearn.over_sampling import SMOTE

# Import the new functions we created
from utils.contrastive_learning import train_contrastive_model, get_prototypes
import seaborn as sns
from scipy.stats import mannwhitneyu


from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer
)

warnings.filterwarnings("ignore", category=ConvergenceWarning)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

power_transformer_pipeline = Pipeline([
    ('robust_scaler', RobustScaler()),  # Step 1: Pre-scale data to a [0, 1] range
    ('power_transform', PowerTransformer(method="yeo-johnson")) # Step 2: Apply the PowerTransformer
])

# list of scalers to sweep
SCALERS = [
    StandardScaler(),                          # z-score (baseline)
    MinMaxScaler(),                            # [0, 1] range
    RobustScaler(),  
    power_transformer_pipeline
    # median / IQR
    # PowerTransformer(method="yeo-johnson", standardize=True)     # de-skew + z-score
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
    
def generate_feature_names(levels=3, stats=("mean", "std", "energy", "entropy", "skew", "kurtosis")):
    """
    Generates descriptive names for each wavelet feature in the vector.
    NOTE: Ensure the 'stats' tuple matches the one in your feature extractor.
    """
    names = []
    # Approximation band name
    for stat in stats:
        names.append(f"LL{levels}_{stat}")
        
    # Detail bands names, from highest level to lowest
    for level in range(levels, 0, -1):
        for band_type in ["LH", "HL", "HH"]:
            for stat in stats:
                names.append(f"{band_type}{level}_{stat}")
    return names

def plot_wavelet_feature_boxplots(df, output_dir="results/handcrafted/wavelet_feature_boxplots"):
    """
    Generates and saves a boxplot for each feature in the DataFrame,
    styled to match your example image.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    feature_columns = [col for col in df.columns if col not in ['label', 'Classe']]
    
    print(f"\nGenerating and saving {len(feature_columns)} boxplots for wavelet features...")
    for i, feature_name in enumerate(feature_columns):
        if (i + 1) % 10 == 0:
            print(f"  Generated {i+1}/{len(feature_columns)} plots...")
            
        plt.figure(figsize=(10, 7))
        
        # Use a color palette that matches your example
        palette = {"C": "#69b3a2", "CRC": "#ff9f80"}
        
        # Create a boxplot and stripplot for rich visualization
        sns.boxplot(data=df, x='Classe', y=feature_name, palette=palette, width=0.5)
        sns.stripplot(data=df, x='Classe', y=feature_name, color=".25", size=2, alpha=0.2)
        
        plt.title(f'Distribution of {feature_name.replace("_", " ").title()}', fontsize=16)
        plt.xlabel('Class', fontsize=14)
        plt.ylabel(feature_name.replace('_', ' ').title(), fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.6)
        
        save_path = os.path.join(output_dir, f'boxplot_{feature_name}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
    print(f"\nAll boxplots have been saved to '{output_dir}'.")
    
    # (Place this function in your analysis script, replacing the old plotting function)

def plot_statistic_grid(df, statistic_name, output_dir="results/wavelet_feature_grids"):
    """
    Generates and saves a grid of boxplots for a single statistic
    (e.g., 'energy') across all 10 wavelet sub-bands.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Find all feature columns related to the chosen statistic
    feature_columns = [col for col in df.columns if col.endswith(f'_{statistic_name}')]
    
    # 2. Define a logical order for plotting the sub-bands
    # LL -> Level 3 (coarsest) -> Level 2 -> Level 1 (finest)
    band_order = ['LL3'] + [f'{direction}{level}' for level in range(3, 0, -1) for direction in ['LH', 'HL', 'HH']]
    ordered_features = [f'{band}_{statistic_name}' for band in band_order]

    # 3. Create the subplot grid
    # A 3x4 grid is spacious enough for 10 plots
    fig, axes = plt.subplots(3, 4, figsize=(20, 15), constrained_layout=True)
    axes = axes.flatten() # Flatten the 2D array of axes for easy iteration

    fig.suptitle(f'Distribution of Wavelet {statistic_name.title()} Features by Class', fontsize=24, weight='bold')

    palette = {"C": "#69b3a2", "CRC": "#ff9f80"}

    # 4. Loop through the ordered features and plot on each subplot
    for i, feature_name in enumerate(ordered_features):
        ax = axes[i]
        sns.boxplot(data=df, x='Classe', y=feature_name, ax=ax, palette=palette, width=0.6)
        
        # Add a title to each subplot indicating the sub-band
        subplot_title = feature_name.split('_')[0]
        ax.set_title(subplot_title, fontsize=16)
        
        ax.set_xlabel('') # Remove individual x-labels
        ax.set_ylabel('') # Remove individual y-labels
        ax.grid(True, linestyle='--', alpha=0.6)

    # 5. Clean up empty subplots
    for i in range(len(ordered_features), len(axes)):
        axes[i].set_visible(False)

    # 6. Add common labels
    fig.supxlabel('Class', fontsize=18)
    fig.supylabel(f'{statistic_name.title()} Value', fontsize=18)
    
    # 7. Save the entire figure
    save_path = os.path.join(output_dir, f'grid_plot_{statistic_name}.png')
    plt.savefig(save_path, dpi=200)
    plt.close()
    
    print(f"Saved grid plot for '{statistic_name}' to '{save_path}'.")
    
def plot_statistical_significance(df, n_top_features=20, output_dir="results/handcrafted"):
    """
    Performs a Mann-Whitney U test for each feature and plots the significance.
    """
    print("\nGenerating Statistical Significance plot...")
    
    # 1. Prepare data for each class
    class_c = df[df['Classe'] == 'C']
    class_crc = df[df['Classe'] == 'CRC']
    
    feature_columns = [col for col in df.columns if col not in ['label', 'Classe']]
    
    p_values = []
    # 2. Loop through each feature and perform the test
    for feature in feature_columns:
        stat, p_val = mannwhitneyu(class_c[feature].dropna(), class_crc[feature].dropna())
        p_values.append({'feature': feature, 'p_value': p_val})
        
    # 3. Create a DataFrame and calculate -log10(p-value)
    # The -log10 transform makes highly significant (small) p-values appear as tall bars
    p_values_df = pd.DataFrame(p_values)
    p_values_df['-log10(p-value)'] = -np.log10(p_values_df['p_value'])
    p_values_df = p_values_df.sort_values('-log10(p-value)', ascending=False).head(n_top_features)

    # 4. Create and save the plot
    plt.figure(figsize=(12, 10))
    sns.barplot(x='-log10(p-value)', y='feature', data=p_values_df, palette='plasma')
    
    # Add a dashed line for the common significance threshold (p=0.05)
    significance_threshold = -np.log10(0.05)
    plt.axvline(x=significance_threshold, color='red', linestyle='--', linewidth=2, label='p = 0.05 threshold')
    
    plt.title(f'Top {n_top_features} Most Statistically Significant Features', fontsize=18)
    plt.xlabel('-log10(p-value)', fontsize=14)
    plt.ylabel('Feature Name', fontsize=14)
    plt.legend()
    plt.grid(True, axis='x', linestyle='--', alpha=0.6)

    save_path = os.path.join(output_dir, 'statistical_significance.png')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()

    print(f"Statistical significance plot saved to '{save_path}'.")

def main():
    # 1) Load data (hold-out split already group-aware)
    train_imgs, train_lbl, train_grp, \
    test_imgs,  test_lbl,  test_grp = load_data()

    handcrafted_X_train = None
    handcrafted_X_test = None
    
    results = {}
    # 2) One pass per feature extractor
    for model_name, model_key in MODELS.items():
        score_dict = {}
        print(f"\n=== Processing extractor: {model_name} ===")

        # 2a) Extract features
        if model_key == "handcrafted":            
            train_cache = os.path.join(CACHE_DIR, "cache_train.joblib")
            test_cache  = os.path.join(CACHE_DIR, "cache_test.joblib")

            X_train_orig = load_or_extract(train_imgs, train_cache,
                                    extract_handcrafted_features)
            X_test_orig  = load_or_extract(test_imgs,  test_cache,
                                    extract_handcrafted_features)
        else: # CNN-based extractor
            X_train_orig = extract_cnn_features(train_imgs, model_key)
            X_test_orig  = extract_cnn_features(test_imgs,  model_key)
        
        handcrafted_X_train = X_train_orig
        handcrafted_X_test = X_test_orig
        #print(X_train)
        # X_train_list = np.nan_to_num(X_train).tolist()

        # pre_scaler = RobustScaler()
        # X_train = pre_scaler.fit_transform(X_train_orig)
        # X_test = pre_scaler.transform(X_test_orig)
        
        # ================================================================= #
        # NEW STEP: APPLY CONTRASTIVE DISSIMILARITY
        # ================================================================= #
        
        # # 1. Train the contrastive model on your original training features
        # contrastive_model = train_contrastive_model(X_train_orig, train_lbl)

        # # 2. Select prototypes from the training set [cite: 176, 338]
        # prototypes, _ = get_prototypes(X_train_orig, train_lbl, n_prototypes_per_class=5)
        
        # # 3. Create the Dissimilarity Space representation
        # # For each sample, calculate its dissimilarity to every prototype
        # print("Creating dissimilarity space for TRAIN set...")
        # X_train_ds = np.array([
        #     contrastive_model.predict(np.abs(X_train_orig - p), verbose=0).flatten()
        #     for p in prototypes
        # ]).T
        
        # def normalize(x, eps=1e-8):
        #     return (x - np.mean(x, axis=1, keepdims=True)) / (np.std(x, axis=1, keepdims=True) + eps)

        # print("Creating dissimilarity space for TEST set...")
        # X_test_ds = np.array([
        #     contrastive_model.predict(normalize(np.abs(X_test_orig - p)), verbose=0).flatten()
        #     for p in prototypes
        # ]).T
        
        # # 4. Combine original features with new dissimilarity features
        # print("Combining original features with dissimilarity space features...")
        
        
        # scaler_orig = RobustScaler()
        # scaler_ds   = RobustScaler()

        # X_train_orig_scaled = scaler_orig.fit_transform(X_train_orig)
        # X_train_ds_scaled   = scaler_ds.fit_transform(X_train_ds)
        
        # # Transformação nos dados de teste (sem re-treinar os scalers!)
        # X_test_orig_scaled = scaler_orig.transform(X_test_orig)
        # X_test_ds_scaled   = scaler_ds.transform(X_test_ds)

        # # Combina os features
        # X_train = np.hstack([X_train_orig_scaled, X_train_ds_scaled])
        # X_test  = np.hstack([X_test_orig_scaled,  X_test_ds_scaled])
        
        # # Reescala o vetor combinado
        # scaler_final = StandardScaler()
        # X_train = scaler_final.fit_transform(X_train).astype(np.float64)
        # X_test  = scaler_final.transform(X_test).astype(np.float64)
        # print(f"New combined feature shape: {X_train.shape}")
        
        # print("Max value:", np.max(X_train))
        # print("Min value:", np.min(X_train))
        # print("Any NaNs:", np.isnan(X_train).any())
        # print("Any infs:", np.isinf(X_train).any())
        
        # print("TEST SET:")
        # print("  Max:", np.max(X_test))
        # print("  Min:", np.min(X_test))
        # print("  Any NaNs:", np.isnan(X_test).any())
        # print("  Any infs:", np.isinf(X_test).any())
        # print("  Any abs > 1e3:", np.any(np.abs(X_test) > 1e3))
        # ================================================================= #
        # END OF NEW STEP
        # ================================================================= #
        
        
        # with open("X_train_features.json", "w") as f:
        #     json.dump(X_train_list, f, indent=2)

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
                    ('variance_remover', VarianceThreshold()), # Step 1: Remove zero-variance features
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
                    cv_inner.split(X_train_orig, train_lbl, train_grp), start=1):
                tr_cnt = np.bincount(train_lbl[tr_idx], minlength=2)
                vl_cnt = np.bincount(train_lbl[val_idx],  minlength=2)
                print(f" Fold {fold}: "
                    f" train → C:{tr_cnt[0]} CRC:{tr_cnt[1]};"
                    f" val → C:{vl_cnt[0]} CRC:{vl_cnt[1]}")
            # ---------------------------------------------
            
            grid = GridSearchCV(
                pipe, param_grid,
                cv=cv_inner, scoring='balanced_accuracy',
                n_jobs=-1, refit=True
            )
            grid.fit(X_train_orig, train_lbl, groups=train_grp)
            
            # # --------------------- NEW CODE BLOCK ---------------------
            # # Extract and print the scores for each fold for the best model
            # print(f"--- Scores per fold for best {clf_name} ---")
            
            # # Find the index of the best performing parameter set
            # best_model_index = grid.best_index_
            
            # # Create a list to store the scores
            # fold_scores = []
            
            # # Loop through each fold (split) and get the score
            # for i in range(cv_inner.get_n_splits()):
            #     fold_score_key = f"split{i}_test_score"
            #     # Get the score of the best model on this specific fold
            #     score = grid.cv_results_[fold_score_key][best_model_index]
            #     fold_scores.append(score)
            #     print(f"  Fold {i+1}: {score:.4f}")
            
            # # Now you have the list of scores for the Wilcoxon test
            # print(f"List of scores for {clf_name}: {np.round(fold_scores, 4)}")
            
            # # You can also add this list to your results dictionary
            # if 'fold_scores' not in results[model_name]:
            #     results[model_name]['fold_scores'] = {}
            # results[model_name]['fold_scores'][clf_name] = fold_scores
            # ------------------- END NEW CODE BLOCK -------------------
            
            
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
            y_pred = grid.best_estimator_.predict(X_test_orig)
            y_score = grid.best_estimator_.predict_proba(X_test_orig)[:, 1]   # ← probability for class 1
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
            }
            
            # plot data distributions
            X_all   = np.vstack([X_train_orig, X_test_orig])
            y_all   = np.hstack([train_lbl, test_lbl])
            scaler = RobustScaler()
            X_scaled = scaler.fit_transform(X_all)
            #X_test = pre_scaler.transform(X_test_orig)
        
            # scaler = PowerTransformer(method='yeo-johnson')    
            # X_scaled  = scaler.fit_transform(X_all)

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

        # 2d) Ensembles (only if we have any base learners)
        if not best_pipelines:
            print(f"No base learners for {model_name}, skipping ensembles.")
        else:
            chosen  = ['Logistic Regression', 'SVM',  'Random Forest', 'MLP2', 'MLP', 'GaussianNB']
            weights = [12, 10, 8, 6, 4]

            # ---- Calibrators (SVM precisa; RF opcional) ----
            svm_cal = CalibratedClassifierCV(best_pipelines['SVM'],  cv=3, method='sigmoid')
            rf_cal  = CalibratedClassifierCV(best_pipelines['Random Forest'], cv=3, method='sigmoid')

            estimators = [
                ('MLP',best_pipelines['MLP']),
                ('SVM', svm_cal),
                ('MLP2',best_pipelines['MLP2']),

                ('Logistic Regression',  best_pipelines['Logistic Regression']),
                ('GaussianNB',  best_pipelines['GaussianNB']),
                #('rf',  rf_cal),

            ]

            soft_vote = VotingClassifier(
                estimators=estimators,
                voting='soft',
                weights=weights,
                n_jobs=-1
            )
            
                        # --- NEW: Get Cross-Validation Scores for Soft Voting Ensemble ---
            print("\n--- Calculating cross-validation scores for Soft Voting ensemble ---")
            
            # 1. Define the cross-validation strategy (the same one used in GridSearchCV)
            # This ensures the comparison is fair
            cv_strategy = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=42)
            
            # 2. Use cross_val_score to get the score for each fold
            # This function will train and evaluate the soft_vote model 4 times, once for each fold.
            # from sklearn.model_selection import cross_val_score
            
            fold_scores = cross_val_score(
                estimator=soft_vote,
                X=X_train_orig,
                y=train_lbl,
                groups=train_grp,
                cv=cv_strategy,
                scoring='balanced_accuracy', # Use the same scoring as your GridSearchCV
                n_jobs=-1
            )
            
            # 3. Print and store the results
            print(f"Soft Voting scores per fold: {np.round(fold_scores, 4)}")
            print(f"Mean CV Balanced Accuracy: {np.mean(fold_scores):.4f}")
            
            # Add these scores to your results dictionary for later use in the Wilcoxon test
            # results[model_name]['SoftVoting']['fold_scores'] = 
            # --------------------- END OF NEW BLOCK ---------------------
            
            
            soft_vote.fit(X_train_orig, train_lbl)
            y_soft = soft_vote.predict(X_test_orig)

            results[model_name]['SoftVoting'] = {
                "members"      : chosen,
                "weights"      : dict(zip(chosen, weights)),
                "test_accuracy": f"{accuracy_score(test_lbl, y_soft):.4f}",
                "report"       : classification_report(test_lbl, y_soft, output_dict=True),
                'fold_scores' : fold_scores.tolist()
            }
            
            # Stacking
            lr      = best_pipelines['Logistic Regression']
            nb      = best_pipelines['GaussianNB']
            MLP     = best_pipelines['MLP']
            MLP2     = best_pipelines['MLP2']


            # 2) estimator fixed list
            estimators = [
                 ('mlp', MLP),
                 ('svm', svm_cal),
                 ('lr',  lr),
                 
                 #('rf',  rf_cal),
                 #('nb',  nb)
            ]

            # 3) Defining Meta classifier
            meta_clf = GradientBoostingClassifier(random_state=42)
            # meta_clf = LogisticRegression(random_state=42, max_iter=1000, solver='saga')
            #meta_clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

            # 4) Creating the meta classifer
            stack = StackingClassifier(
                estimators=estimators,
                final_estimator=meta_clf,
                cv=4,                   
                n_jobs=-1,
                passthrough=False        # pass only the probabilities
            )

            # 5) Treine e avalie
            stack.fit(X_train_orig, train_lbl)
            y_pred = stack.predict(X_test_orig)
            acc    = accuracy_score(test_lbl, y_pred)
            print(f"Stacking accuracy: {acc:.4f}")
                        
            # Save confusion matrix for soft voting
            out_dir = os.path.join("results", model_name)
            os.makedirs(out_dir, exist_ok=True)
            plot_and_save_confusion_matrix(
                test_lbl, y_pred,
                title=f"Confusion Matrix – Stacking",
                filename="confmat_stack",
                output_dir=out_dir
            )
            
            # # Save confusion matrix for soft voting
            # out_dir = os.path.join("results", model_name)
            # os.makedirs(out_dir, exist_ok=True)
            # plot_and_save_confusion_matrix(
            #     test_lbl, y_soft,
            #     title=f"Confusion Matrix – Soft Voting",
            #     filename="confmat_softvoting.png.png",
            #     output_dir=out_dir
            # )

            results[model_name]['Stacking'] = {
                "base_learners": 'mlp,svm,lr,rf,nb',
                # "base_learners": 'rf,lr,nb',

                "meta": "GradientBoostingClassifier",
                "test_accuracy": f"{accuracy_score(test_lbl, y_pred):.4f}",
                "report": classification_report(test_lbl, y_pred, output_dict=True)
            }
            
            
    # --- NEW SECTION: Generate Boxplots for Handcrafted Features ---
    # # This block will only run if the handcrafted features were processed in the loop
    # if handcrafted_X_train is not None and handcrafted_X_test is not None:
    #     print("\n=== Starting Feature Analysis for Handcrafted Wavelet Features ===")
        
    #     # 1. Combine train and test sets to visualize the entire dataset
    #     all_features_matrix = np.vstack((handcrafted_X_train, handcrafted_X_test))
    #     all_labels = np.hstack((train_lbl, test_lbl))

    #     # 2. Get feature names
    #     # IMPORTANT: Make sure these parameters match exactly what is inside your
    #     # `extract_handcrafted_features` and `extract_wavelet_features` functions.
    #     # Your code uses wavelet='db5', levels=3 and extracts 6 stats.
    #     feature_names = generate_feature_names(
    #         levels=3, 
    #         stats=("mean", "std", "energy", "entropy")
    #     )
        
    #     # 3. Create a Pandas DataFrame for plotting
    #     if len(feature_names) == all_features_matrix.shape[1]:
    #         df_features = pd.DataFrame(all_features_matrix, columns=feature_names)
    #         df_features['label'] = all_labels
    #         df_features['Classe'] = df_features['label'].map({0: 'C', 1: 'CRC'})

    #         # 4. Generate and save all the boxplots
    #         plot_wavelet_feature_boxplots(df_features)
    #     else:
    #         print("\nWARNING: Mismatch in feature dimensions. Skipping boxplot generation.")
    #         print(f"  Expected {len(feature_names)} features but found {all_features_matrix.shape[1]}.")
    #         print("  Please check the `stats` tuple in `generate_feature_names` and `extract_wavelet_features`.")
    # -----------------------------------------------------------------
# --- NEW SECTION: Generate Summary Analysis Plots for Handcrafted Features ---
    # if handcrafted_X_train is not None and handcrafted_X_test is not None:
    #         print("\n=== Starting Summary Feature Analysis for Handcrafted Wavelet Features ===")
            
    #         all_features_matrix = np.vstack((handcrafted_X_train, handcrafted_X_test))
    #         all_labels = np.hstack((train_lbl, test_lbl))

    #         # IMPORTANT: This tuple must match what your feature extractor produces.
    #         stats_tuple = ("mean", "std", "energy", "entropy")
    #         feature_names = generate_feature_names(levels=3, stats=stats_tuple)
            
    #         if len(feature_names) == all_features_matrix.shape[1]:
    #             # Create the DataFrame once, with all necessary data
    #             df_features = pd.DataFrame(all_features_matrix, columns=feature_names)
    #             df_features['Classe'] = pd.Series(all_labels).map({0: 'C', 1: 'CRC'})
    #             df_features['label'] = pd.Series(all_labels) # Add numeric label for the functions

    #             # --- Call the new single-plot functions here ---
    #             # This replaces the loop for the grid plots to create more compact figures
    #             print("\n--- Generating Summary Analysis Plots ---")
    #             # plot_feature_importance(df_features)
    #             plot_statistical_significance(df_features)
    #             # -----------------------------------------------

    #         else:
    #             print("\nWARNING: Mismatch in feature dimensions. Skipping summary plot generation.")
    #             print(f"  Expected {len(feature_names)} features but found {all_features_matrix.shape[1]}.")

    # # ---------------------------------------------------------------------------------

    # 3) Persist all results
    save_results(results, RESULTS_FILE)


if __name__ == "__main__":
    main()
