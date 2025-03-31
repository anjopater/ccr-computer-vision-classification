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

def plot_and_save_pca_3d(X_pca, labels, title, filename):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')  # Create a 3D plot

    unique_labels = np.unique(labels)
    for label in unique_labels:
        idx = labels == label
        ax.scatter(X_pca[idx, 0], X_pca[idx, 1], X_pca[idx, 2], label=f"Class {label}", alpha=0.7)

    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.set_zlabel("Principal Component 3")
    ax.set_title(title)
    ax.legend()
    ax.grid(True)

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

    train_pca_3d_plot_path = os.path.join(output_dir, "train_pca_plot_3d.png")
    plot_and_save_pca_3d(X_train_pca, train_labels, "PCA Visualization of Training Data", train_pca_3d_plot_path)

    test_pca_plot_path = os.path.join(output_dir, "test_pca_plot.png")
    plot_and_save_pca(X_test_pca, test_labels, "PCA Visualization of Testing Data", test_pca_plot_path)

    test_pca_3d_plot_path = os.path.join(output_dir, "test_pca_plot_3d.png")
    plot_and_save_pca_3d(X_test_pca, test_labels, "PCA Visualization of Testing Data", test_pca_3d_plot_path)


        # UMAP Visualization
    reducer = umap.UMAP(a=None, angular_rp_forest=False, b=None,
     force_approximation_algorithm=False, init='spectral', learning_rate=1.0,
     local_connectivity=1.0, low_memory=False, metric='euclidean',
     metric_kwds=None, min_dist=0.1, n_components=2, n_epochs=None,
     n_neighbors=15, negative_sample_rate=5, output_metric='euclidean',
     output_metric_kwds=None, random_state=42, repulsion_strength=1.0,
     set_op_mix_ratio=1.0, spread=1.0, target_metric='categorical',
     target_metric_kwds=None, target_n_neighbors=-1, target_weight=0.5,
     transform_queue_size=4.0, transform_seed=42, unique=False, verbose=False)  # Initialize UMAP
    X_train_umap = reducer.fit_transform(X_train_scaled)  # Fit and transform training data
    X_test_umap = reducer.transform(X_test_pca)  # Transform testing data (using PCA as input)

    # Save UMAP plots
    train_umap_plot_path = os.path.join(output_dir, "train_umap_plot.png")
    plot_and_save_umap(X_train_umap, train_labels, "UMAP Visualization of Training Data", train_umap_plot_path)

    test_umap_plot_path = os.path.join(output_dir, "test_umap_plot.png")
    plot_and_save_umap(X_test_umap, test_labels, "UMAP Visualization of Testing Data", test_umap_plot_path)

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
    
    # Compute the confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Calculate percentages
    cm_percentage = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    
    # Annotate with both counts and percentages
    annot = []
    for i in range(cm.shape[0]):
        row = []
        for j in range(cm.shape[1]):
            count = cm[i, j]
            percentage = cm_percentage[i, j]
            row.append(f"{count}\n({percentage:.2f}%)")
        annot.append(row)
    
    sns.heatmap(cm, annot=annot, fmt="", cmap="Blues", cbar=False,
                xticklabels=["Control", "Carcinogenesis"],
                yticklabels=["Control", "Carcinogenesis"], 
                annot_kws={"size": 12}, cbar_kws={"shrink": 0.8})
    
    # Title and labels
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    
    # Save the plot
    plt.savefig(filename)
    plt.close()

def plot_cv_indices(cv, X, y, group, n_splits, output_dir):
    """
    Plot the indices of the cross-validation splits and log them in text format.
    Also log the unique animals (groups) in each fold.
    
    Parameters:
    - cv: Cross-validation object (e.g., GroupKFold).
    - X: Feature matrix.
    - y: Target labels.
    - group: Group labels.
    - n_splits: Number of splits.
    - output_dir: Directory to save the plot and log file.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(12, 6))

    # Open a log file to save the CV splits
    log_file_path = os.path.join(output_dir, "cv_splits_log.txt")
    with open(log_file_path, "w") as log_file:
        # Generate the splits
        for i, (train, test) in enumerate(cv.split(X, y, group)):
            # Log the training and testing indices
            log_file.write(f"Fold {i+1}:\n")
            log_file.write(f"  Training indices: {train}\n")
            log_file.write(f"  Testing indices: {test}\n")
            
            # Log the unique animals (groups) in the training and testing sets
            train_groups = group[train]
            test_groups = group[test]
            log_file.write(f"  Training animals: {np.unique(train_groups)}\n")
            log_file.write(f"  Testing animals: {np.unique(test_groups)}\n")
            log_file.write("\n")

            # Create an array to represent the indices
            indices = np.array([np.nan] * len(X))
            indices[train] = 1  # Training set
            indices[test] = 0   # Test set

            # Set colors based on the indices
            colors = ['orange' if i == 1 else 'blue' for i in indices]

            # Plot the indices with custom colors (orange for train, blue for test)
            ax.scatter(range(len(indices)), [i + 0.5] * len(indices), 
                       c=colors, marker='_', lw=10)
            print(f"Fold {i+1}:")
            print(f"  Training animals: {np.unique(train_groups)}")
            print(f"  Testing animals: {np.unique(test_groups)}")

    # Customize the plot
    ax.set_yticks(np.arange(n_splits) + 0.5)
    ax.set_yticklabels([f"Fold {i+1}" for i in range(n_splits)])
    ax.set_xlabel("Index")
    ax.set_ylabel("CV Fold")
    ax.set_title("Cross-Validation Indices for GroupKFold")

    # Create custom legend
    from matplotlib.patches import Patch
    ax.legend([Patch(color='orange'), Patch(color='blue')], ['Train Set', 'Test Set'], loc='upper right')

    # Save the plot
    cv_plot_path = os.path.join(output_dir, "cv_indices_plot.png")
    plt.savefig(cv_plot_path)
    plt.close()

    print(f"CV indices plot saved to: {cv_plot_path}")
    print(f"CV splits log saved to: {log_file_path}")

    return cv_plot_path, log_file_path



def plot_and_save_feature_histograms(features, labels, class_names, output_dir, num_bins=30):
    """
    Plots and saves histograms of extracted features, separated by class.

    :param features: numpy array of extracted features (shape: [num_samples, num_features])
    :param labels: numpy array of class labels (same length as features)
    :param class_names: list of class names (e.g., ["Control", "Cancer"])
    :param output_dir: directory to save the plots
    :param num_bins: number of bins in the histogram
    """
    os.makedirs(output_dir, exist_ok=True)
    
    num_features = features.shape[1]

    for i in range(num_features):
        plt.figure(figsize=(8, 6))
        
        # Plot histogram for each class
        for class_idx, class_name in enumerate(class_names):
            sns.histplot(features[labels == class_idx, i], bins=num_bins, kde=True, label=class_name, alpha=0.6)

        # Plot formatting
        plt.title(f"Feature {i+1} Distribution")
        plt.xlabel(f"Feature {i+1} Value")
        plt.ylabel("Count")
        plt.legend()
        
        # Save the plot
        filename = os.path.join(output_dir, f"feature_{i+1}_histogram.png")
        plt.savefig(filename)
        plt.close()


def plot_and_save_svm_kernel(X, y_true, y_pred, title, filename, output_dir, model):
    """
    Plots the decision boundary of a trained SVM model and overlays true vs. predicted labels.
    
    Parameters:
        X (array-like): Feature matrix (2D).
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels.
        title (str): Title of the plot.
        filename (str): Name of the output file.
        output_dir (str): Directory to save the plot.
        model (sklearn model): Trained SVM model.
    """
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot decision boundary
    plt.figure(figsize=(8, 6))
    print("PLOTTING......")
    DecisionBoundaryDisplay.from_estimator(
        model,
        X,
        response_method="predict",
        cmap=plt.cm.Spectral,
        alpha=0.6,
        xlabel="Feature 1",
        ylabel="Feature 2",
        grid_resolution=50  # Reduce resolution for faster plotting

    )
    
    # Scatter plot: True vs. Predicted labels
    plt.scatter(X[:, 0], X[:, 1], c=y_true, s=30, edgecolors="k", label="True Labels", marker="o")
    plt.scatter(X[:, 0], X[:, 1], c=y_pred, s=10, edgecolors="r", label="Predicted Labels", marker="x")

    plt.title(title)
    plt.legend()
    
    # Save plot
    plot_path = os.path.join(output_dir, filename)
    plt.savefig(plot_path)
    plt.close()
    
    print(f"Plot saved at: {plot_path}")