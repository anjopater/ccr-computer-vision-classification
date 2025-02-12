# main.py
from config import PCA_COMPONENTS, MODELS, RESULTS_FILE
from utils.data_loader import load_data
from utils.feature_extractor import extract_cnn_features, apply_pca
from utils.evaluator import train_and_evaluate
from utils.logger import save_results
from utils.save_plots import plot_and_save_pca, plot_and_save_explained_variance, plot_and_save_correlation_heatmap, generate_plots_and_save_results, plot_and_save_feature_histograms
import os
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import StandardScaler
import pandas as pd


# Disable warnings
warnings.filterwarnings("ignore")  # Disable all warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)  # Suppress scikit-learn ConvergenceWarning
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings
warnings.simplefilter("ignore", ConvergenceWarning)


def main():
    results = {}
    train_images, train_labels, train_groups, test_images, test_labels, test_groups = load_data()
    print(f"Number of training images: {len(train_images)}")
    print(f"Number of training labels: {len(train_labels)}")
    print(f"Number of test images: {len(test_images)}")
    print(f"Number of test labels: {len(test_labels)}")

    for model_name, model_key in MODELS.items():
        results[model_name] = {}
        train_features = extract_cnn_features(train_images, model_key)
        test_features = extract_cnn_features(test_images, model_key)

        print(f"Shape of train_features: {train_features.shape}")
        print(f"Shape of test_features: {test_features.shape}")
        # print(train_features) 

        for n_components in PCA_COMPONENTS:
            print(f"Testing {model_name} with {n_components} PCA components...")
            print(train_features.shape)
            print(test_features.shape)
            train_features_pca, test_features_pca, pca, scaler = apply_pca(train_features, test_features, n_components)
            # plot_and_save_feature_histograms(train_features_pca, train_labels, ["Control", "Cancer"], output_dir="feature_histograms")

            print(f"Shape of train_features_pca: {train_features_pca.shape}")
            print(f"Shape of test_features_pca: {test_features_pca.shape}")

            classifier_results = train_and_evaluate(train_features_pca, train_labels, test_features_pca, test_labels, train_groups, test_groups, model_name, n_components)
            results[model_name][n_components] = classifier_results

            # Generate plots and save them
            output_dir = os.path.join("results", model_name, f"pca_{n_components}")
            plot_paths = generate_plots_and_save_results(
                train_features, train_features_pca, train_labels,
                test_features_pca, test_labels, pca, output_dir
            )

            # Train and evaluate the model
            # classifier_results = train_and_evaluate(
            #     train_features_pca, train_labels, test_features_pca, test_labels, train_groups
            # )

            # Add plot paths to the results
            # classifier_results["plots"] = plot_paths

    save_results(results, RESULTS_FILE)

if __name__ == "__main__":
    main()