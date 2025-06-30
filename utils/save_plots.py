import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.metrics import confusion_matrix
import umap
from matplotlib.patches import Patch
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.metrics import roc_curve, auc
from sklearn.manifold import SpectralEmbedding   # Laplacian Eigenmaps
from sklearn.manifold._t_sne import TSNE   # caminho interno explícito


def plot_and_save_pca(X_pca, labels, title, filename):
    plt.figure(figsize=(10, 7))
    unique_labels = np.unique(labels)
    for label in unique_labels:
        idx = labels == label
        plt.scatter(X_pca[idx, 0], X_pca[idx, 1], label=f"Class {label}", alpha=0.7)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)  # Save the plot
    plt.close()  # Close the plot to free memory


def plot_and_save_umap(X_umap, labels, title, filename):
    plt.figure(figsize=(10, 7))
    unique_labels = np.unique(labels)
    for label in unique_labels:
        idx = labels == label
        plt.scatter(X_umap[idx, 0], X_umap[idx, 1], label=f"Class {label}", alpha=0.7)
    plt.xlabel("UMAP Component 1")
    plt.ylabel("UMAP Component 2")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)  # Save the plot
    plt.close()  # Close the plot to free memory
    
def compute_tsne(
        X,
        n_components: int = 2,
        perplexity: float = 30.0,
        learning_rate: float | str = "auto",
        random_state: int = 42,
    ) -> np.ndarray:

    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        learning_rate=learning_rate,
        init="pca",
        random_state=random_state
    )
    return tsne.fit_transform(X)
    
def plot_and_save_confusion_matrix(y_true, y_pred, title, filename, output_dir):
    """
    Plot & save a confusion matrix with both counts and row percentages.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Compute counts
    cm = confusion_matrix(y_true, y_pred)
    # Compute row‐normalized percentages
    cm_pct = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis] * 100

    # Build annotation strings
    annot = np.empty_like(cm).astype(object)
    n_classes = cm.shape[0]
    for i in range(n_classes):
        for j in range(n_classes):
            c = cm[i, j]
            p = cm_pct[i, j]
            annot[i, j] = f"{c}\n({p:.1f}%)"

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=annot,
        fmt="",
        cmap="Blues",
        cbar=False,
        xticklabels=["Control","Carcinogenesis"],
        yticklabels=["Control","Carcinogenesis"],
        annot_kws={"size":12}
    )
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")

    out_path = os.path.join(output_dir, filename)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved confusion matrix to {out_path}")

def plot_cv_indices(cv, X, y, group, n_splits, output_dir):
    import os, numpy as np, matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 6))

    # Build a mapping animal → list of positions
    positions = {}
    for idx, animal in enumerate(group):
        positions.setdefault(animal, []).append(idx)

    # Plot CV splits as before
    for fold_idx, (train, test) in enumerate(cv.split(X, y, group)):
        indices = np.full(len(X), np.nan)
        indices[train] = 1
        indices[test]  = 0
        colors = ['orange' if v == 1 else 'blue' for v in indices]
        ax.scatter(range(len(X)), [fold_idx + 0.5]*len(X),
                   c=colors, marker='_', lw=10)

    # Now set x‐ticks at each animal’s median position
    animal_ids   = list(positions.keys())
    tick_locs    = [np.median(positions[a]) for a in animal_ids]
    ax.set_xticks(tick_locs)
    ax.set_xticklabels(animal_ids, rotation=0, fontsize=10)

    # Y labels
    ax.set_yticks(np.arange(n_splits) + 0.5)
    ax.set_yticklabels([f"Fold {i+1}" for i in range(n_splits)])

    ax.set_xlabel("Animal ID")
    ax.set_ylabel("CV Fold")
    ax.set_title("Cross-Validation Splits by Animal")

    # Legend
    ax.legend([Patch(color='orange'), Patch(color='blue')],
              ['Train','Validation'],
              loc='upper right')

    # Save everything
    log_file = os.path.join(output_dir, "cv_splits_log.txt")
    # …write the same logs if you like…
    plot_path = os.path.join(output_dir, "cv_indices_labeled.png")
    fig.savefig(plot_path, bbox_inches="tight")
    plt.close(fig)

    return plot_path, log_file


def plot_fold_animal_heatmap(cv, groups, labels, output_dir):
    """
    Plot a clean heatmap where rows = CV folds, cols = animal IDs,
    and cells indicate Train (orange) vs. Validation (blue).
    
    Parameters
    ----------
    cv : cross-validation splitter
        Any sklearn splitter with .split(..., groups=groups) and .n_splits attribute.
    groups : array-like of shape (n_samples,)
        Group labels for each sample (e.g. "C_1", "CRC_3", etc.).
    labels : array-like of shape (n_samples,)
        True class labels (only used to pass into cv.split).
    output_dir : str
        Directory where the heatmap PNG will be saved.
    
    Saves
    -----
    cv_animal_heatmap.png in output_dir
    """
    n_splits = cv.n_splits
    os.makedirs(output_dir, exist_ok=True)

    # Determine unique animals in order of first appearance
    animals = []
    seen = set()
    for g in groups:
        if g not in seen:
            seen.add(g)
            animals.append(g)

    # Build matrix: 0 = train, 1 = val
    mat = np.zeros((n_splits, len(animals)), dtype=int)
    for fold_idx, (_, val_idx) in enumerate(
            cv.split(np.zeros(len(groups)), labels, groups)):
        val_animals = set(groups[val_idx])
        for j, a in enumerate(animals):
            if a in val_animals:
                mat[fold_idx, j] = 1

    # Plot it
    plt.figure(figsize=(len(animals)*0.5 + 2, n_splits*0.6 + 1))
    sns.set_style("white")

    ax = sns.heatmap(
        mat,
        cmap=['#FFA500', '#1f77b4'],  # train=orange, val=blue
        cbar=False,
        linewidths=0.5,
        linecolor="white",
        xticklabels=animals,
        yticklabels=[f"Fold {i+1}" for i in range(n_splits)],
        square=True
    )

    ax.set_xlabel("Animal ID", fontsize=10)
    ax.set_ylabel("CV Fold",   fontsize=10)
    ax.set_title("Train vs. Validation Animals per Fold", fontsize=12, pad=12)

    legend_patches = [
        Patch(facecolor='#FFA500', edgecolor='white', label='Train'),
        Patch(facecolor='#1f77b4', edgecolor='white', label='Validation'),
    ]
    ax.legend(
            handles   = legend_patches,
            loc       = 'upper center',
            bbox_to_anchor = (0.5, -0.15),  # y-offset below the axes
            ncol      = 2,
            frameon   = False
        )

    plt.tight_layout()
    out_path = os.path.join(output_dir, "cv_animal_group_heatmap.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved CV animal-heatmap to {out_path}")
    return out_path

def plot_multiple_roc(y_true, score_dict, title, filename, output_dir):
    """
    Draw one ROC plot with many curves.

    Parameters
    ----------
    y_true : array-like shape (n_samples,)
        Ground-truth binary labels (0/1).
    score_dict : dict[str, array-like]
        Model name → positive-class probability for each sample.
    """
    plt.figure(figsize=(5, 5))

    # diagonal
    plt.plot([0, 1], [0, 1], ls="--", lw=1, label="Chance")

    for name, scores in score_dict.items():
        fpr, tpr, _ = roc_curve(y_true, scores)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=1.8, label=f"{name} (AUC {roc_auc:.3f})")

    plt.xlabel("False-Positive Rate")
    plt.ylabel("True-Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right", fontsize="x-small")
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    plt.close()