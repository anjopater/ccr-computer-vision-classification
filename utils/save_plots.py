import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

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

    test_pca_plot_path = os.path.join(output_dir, "test_pca_plot.png")
    plot_and_save_pca(X_test_pca, test_labels, "PCA Visualization of Testing Data", test_pca_plot_path)

    # Save explained variance plot
    explained_variance_plot_path = os.path.join(output_dir, "explained_variance_plot.png")
    plot_and_save_explained_variance(pca, explained_variance_plot_path)

    # Save correlation heatmap
    correlation_heatmap_path = os.path.join(output_dir, "correlation_heatmap.png")
    plot_and_save_correlation_heatmap(X_train_scaled, correlation_heatmap_path)

    # Return file paths for JSON
    return {
        "train_pca_plot": train_pca_plot_path,
        "test_pca_plot": test_pca_plot_path,
        "explained_variance_plot": explained_variance_plot_path,
        "correlation_heatmap": correlation_heatmap_path
    }

def plot_and_save_confusion_matrix(y_true, y_pred, title, filename, output_dir):
     # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Control", "Cancer"],
                yticklabels=["Control", "Cancer"])
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(filename)
    plt.close()
