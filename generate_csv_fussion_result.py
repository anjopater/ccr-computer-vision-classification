
import json
import csv

# Load your JSON data
# Load JSON from a file
with open("results.json", "r") as file:
    data = json.load(file) 

# Define CSV headers
headers = [
    "Extractor_PCA_Fusion",  # Combined identifier\
    "Fussion",
    "Accuracy",
    "Precision",
    "Recall",
    "F1-Score"
]

rows = []

# Iterate through extractors
for extractor, components_data in data.items():
    # Iterate through PCA components
    for components, classifiers in components_data.items():
        # Check if Fusion exists in classifiers
        if "Fusion" in classifiers:
            fusion_data = classifiers["Fusion"]
            # Iterate through fusion types
            for fusion_type, fusion_details in fusion_data.items():
                # Create combined identifier
                combined_id = f"{extractor}_{components}_{fusion_type.replace(' ', '_')}"
                
                # Extract metrics
                accuracy = fusion_details["accuracy"]
                report = fusion_details["report"]["macro avg"]
                
                rows.append([
                    combined_id,
                    fusion_type,
                    accuracy,
                    report["precision"],
                    report["recall"],
                    report["f1-score"]
                ])

# Write to CSV
with open("fusion_results.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(headers)
    writer.writerows(rows)

print("CSV file created successfully!")