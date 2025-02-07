import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import umap
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# def plot_and_save_pca(X_pca, labels, title, filename):
#     plt.figure(figsize=(10, 7))
#     unique_labels = np.unique(labels)
#     for label in unique_labels:
#         idx = labels == label
#         plt.scatter(X_pca[idx, 0], X_pca[idx, 1], label=f"Class {label}", alpha=0.7)
#     plt.xlabel("Principal Component 1")
#     plt.ylabel("Principal Component 2")
#     plt.title(title)
#     plt.legend()
#     plt.grid(True)
#     plt.savefig(filename)  # Save the plot
#     plt.close()  # Close the plot to free memory

# def plot_and_save_pca_3d(X_pca, labels, title, filename):
#     fig = plt.figure(figsize=(10, 7))
#     ax = fig.add_subplot(111, projection='3d')  # Create a 3D plot

#     unique_labels = np.unique(labels)
#     for label in unique_labels:
#         idx = labels == label
#         ax.scatter(X_pca[idx, 0], X_pca[idx, 1], X_pca[idx, 2], label=f"Class {label}", alpha=0.7)

#     ax.set_xlabel("Principal Component 1")
#     ax.set_ylabel("Principal Component 2")
#     ax.set_zlabel("Principal Component 3")
#     ax.set_title(title)
#     ax.legend()
#     ax.grid(True)

#     plt.savefig(filename)  # Save the plot
#     plt.close()  # Close the plot to free memory

def plot_and_save_explained_variance(pca, filename):
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='--')
    plt.axhline(y=0.90, color='r', linestyle='-')  # 90% variance threshold
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('PCA Explained Variance')
    plt.grid()
    plt.savefig(filename)  # Save the plot
    plt.close()  # Close the plot to free memory

def plot_and_save_umap(X_umap, labels, title, filename):
    plt.figure(figsize=(10, 7))
    unique_labels = np.unique(labels)
    for label in unique_labels:
        idx = labels == label
        plt.scatter(X_umap[idx, 0], X_umap[idx, 1], label=f"Class {label}", alpha=0.7)
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)  # Save the plot
    plt.close()  # Close the plot to free memory

def plot_and_save_correlation_heatmap(X_train, filename):
    plt.figure(figsize=(12, 6))
    sns.heatmap(np.corrcoef(X_train.T), cmap="coolwarm", center=0)
    plt.title("Feature Correlation Heatmap")
    plt.savefig(filename)  # Save the plot
    plt.close()  # Close the plot to free memory

def generate_plots_and_save_results(X_train_scaled, X_train_pca, train_labels, X_test_pca, test_labels, pca, output_dir):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save PCA plots
    train_pca_plot_path = os.path.join(output_dir, "train_pca_plot.png")
    plot_and_save_pca(X_train_pca, train_labels, "PCA Visualization of Training Data", train_pca_plot_path)

    train_pca_3d_plot_path = os.path.join(output_dir, "train_pca_plot_3d.png")
    plot_and_save_pca_3d(X_train_pca, train_labels, "PCA Visualization of Training Data", train_pca_3d_plot_path)

    test_pca_plot_path = os.path.join(output_dir, "test_pca_plot.png")
    plot_and_save_pca(X_test_pca, test_labels, "PCA Visualization of Testing Data", test_pca_plot_path)

    test_pca_3d_plot_path = os.path.join(output_dir, "test_pca_plot_3d.png")
    plot_and_save_pca_3d(X_test_pca, test_labels, "PCA Visualization of Testing Data", test_pca_3d_plot_path)

    # Save explained variance plot
    explained_variance_plot_path = os.path.join(output_dir, "explained_variance_plot.png")
    plot_and_save_explained_variance(pca, explained_variance_plot_path)

    # Save correlation heatmap
    correlation_heatmap_path = os.path.join(output_dir, "correlation_heatmap.png")
    plot_and_save_correlation_heatmap(X_train_scaled, correlation_heatmap_path)

    train_umap_plot_path = os.path.join(output_dir, "train_umap_plot.png")
    plot_and_save_umap(X_train_pca, train_labels, "UMAP Visualization of Training Data", train_umap_plot_path)

    test_umap_plot_path = os.path.join(output_dir, "test_umap_plot.png")
    plot_and_save_umap(X_test_pca, test_labels, "UMAP Visualization of Testing Data", test_umap_plot_path)
    

    # Return file paths for JSON
    return {
        "train_pca_plot": train_pca_plot_path,
        "test_pca_plot": test_pca_plot_path,
        "explained_variance_plot": explained_variance_plot_path,
        "correlation_heatmap": correlation_heatmap_path
    }

def plot_and_save_confusion_matrix(y_true, y_pred, title, filename, output_dir="confusion_matrices"):
    """
    Plots and saves the confusion matrix for all folds combined.
    
    Parameters:
        y_true (array): True labels.
        y_pred (array): Predicted labels.
        title (str): Title of the plot.
        filename (str): Filename for saving the plot.
        output_dir (str): Directory where the plot will be saved.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Compute the confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Calculate percentages
    cm_percentage = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    
    # Annotate with counts and percentages
    annot = [[f"{cm[i, j]}\n({cm_percentage[i, j]:.2f}%)" for j in range(cm.shape[1])] for i in range(cm.shape[0])]
    
    sns.heatmap(cm, annot=annot, fmt="", cmap="Blues", cbar=False,
                xticklabels=np.unique(y_true),
                yticklabels=np.unique(y_true),
                annot_kws={"size": 7}, cbar_kws={"shrink": 0.8})
    
    # Title and labels
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    
    # Save the plot
    plt.savefig(os.path.join(filename))
    plt.close()

def plot_cv_indices(cv, X, y, group, ax, n_splits, lw=10):
    """Create a sample plot for indices of a cross-validation object."""
    use_groups = "Group" in type(cv).__name__
    groups = group if use_groups else None
    # Generate the training/testing visualizations for each CV split
    for ii, (tr, tt) in enumerate(cv.split(X=X, y=y, groups=groups)):
        # Fill in indices with the training/test groups
        indices = np.array([np.nan] * len(X))
        indices[tt] = 1
        indices[tr] = 0

        # Visualize the results
        ax.scatter(
            range(len(indices)),
            [ii + 0.5] * len(indices),
            c=indices,
            marker="_",
            lw=lw,
            cmap=cmap_cv,
            vmin=-0.2,
            vmax=1.2,
        )

    # Plot the data classes and groups at the end
    ax.scatter(
        range(len(X)), [ii + 1.5] * len(X), c=y, marker="_", lw=lw, cmap=cmap_data
    )

    ax.scatter(
        range(len(X)), [ii + 2.5] * len(X), c=group, marker="_", lw=lw, cmap=cmap_data
    )

    # Formatting
    yticklabels = list(range(n_splits)) + ["class", "group"]
    ax.set(
        yticks=np.arange(n_splits + 2) + 0.5,
        yticklabels=yticklabels,
        xlabel="Sample index",
        ylabel="CV iteration",
        ylim=[n_splits + 2.2, -0.2],
        xlim=[0, 100],
    )
    ax.set_title("{}".format(type(cv).__name__), fontsize=15)
    return ax

import matplotlib.pyplot as plt
import numpy as np

def plot_custom_cv(extracted_features):
    """
    Visualizes the custom cross-validation process.
    
    Parameters:
      extracted_features: list of tuples (features, labels) for each fold.
    """
    n_folds = len(extracted_features)
    n_samples = sum(len(features) for features, _ in extracted_features)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, n_folds * 0.5))
    
    # Index to track sample positions
    sample_start = 0

    for fold_idx in range(n_folds):
        features, _ = extracted_features[fold_idx]
        fold_size = len(features)

        # Assign training (gray) and test (blue) indices
        train_indices = np.arange(n_samples)
        test_indices = np.arange(sample_start, sample_start + fold_size)

        # Remove test indices from training indices
        train_indices = np.setdiff1d(train_indices, test_indices)

        # Plot training samples
        ax.scatter(train_indices, [fold_idx] * len(train_indices), 
                   c="gray", marker="_", lw=10, label="Training" if fold_idx == 0 else None)

        # Plot test samples
        ax.scatter(test_indices, [fold_idx] * len(test_indices), 
                   c="blue", marker="_", lw=10, label="Test" if fold_idx == 0 else None)

        # Update sample_start for next fold
        sample_start += fold_size

    ax.set_xlabel("Sample Index")
    ax.set_ylabel("CV Fold")
    ax.set_title("Cross-Validation Splits Visualization")
    ax.legend()
    plt.show()


def plot_and_save_pca(features_pca, labels, title, filename, output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(8, 6))
    
    # Correctly create the scatter plot and colormap
    scatter = plt.scatter(features_pca[:, 0], features_pca[:, 1], c=labels, cmap="viridis", alpha=0.7)

    plt.colorbar(scatter, label="Classes")  # Attach colorbar to scatter
    plt.title(title)
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")

    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"PCA plot saved to: {save_path}")

def plot_and_save_umap(features_umap, labels, title, filename, output_dir):
    """
    Plots and saves UMAP visualization.
    
    Parameters:
      features_umap: Data after applying UMAP transformation.
      labels: Ground truth labels.
      title: Title for the plot.
      filename: Filename for saving the plot.
      output_dir: Directory to save the plot.
    """
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(features_umap[:, 0], features_umap[:, 1], c=labels, cmap='tab10', alpha=0.7)
    plt.title(title)
    plt.colorbar(scatter)
    plt.xlabel('UMAP Component 1')
    plt.ylabel('UMAP Component 2')
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()
