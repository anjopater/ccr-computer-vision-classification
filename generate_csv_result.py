import json
import csv

# Load JSON from a file
with open("results.json", "r") as file:
    data = json.load(file) 
# Load your JSON data

# Define CSV headers (now includes "Classifier")
headers = ["Extractor", "PCA Components", "Classifier", "Best Parameters", "Accuracy", "Precision", "Recall", "F1-Score"]

rows = []

# Iterate through extractors, components, and classifiers (excluding Fusion)
for extractor, components_data in data.items():
    for components, classifiers in components_data.items():
        for classifier, details in classifiers.items():
            # Skip Fusion entries
            if classifier == "Fusion":
                continue
            
            # Extract metrics and parameters
            accuracy = details["accuracy"]
            report = details["report"]
            precision = report["macro avg"]["precision"]
            recall = report["macro avg"]["recall"]
            f1 = report["macro avg"]["f1-score"]
            best_params = str(details["best_params"])
            
            rows.append([
                extractor, 
                components, 
                classifier,  # Added classifier name (e.g., "KNN", "SVM")
                best_params, 
                accuracy, 
                precision, 
                recall, 
                f1
            ])

# Write to CSV
with open("classifier_results_with_names.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(headers)
    writer.writerows(rows)