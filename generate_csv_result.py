# import json
# import csv

# # Load JSON from a file
# with open("results.json", "r") as file:
#     data = json.load(file) 
# # Load your JSON data

# # Define CSV headers (now includes "Classifier")
# headers = ["Extractor", "PCA Components", "Classifier", "Best Parameters", "Accuracy", "Precision", "Recall", "F1-Score"]

# rows = []

# # Iterate through extractors, components, and classifiers (excluding Fusion)
# for extractor, components_data in data.items():
#     for components, classifiers in components_data.items():
#         for classifier, details in classifiers.items():
#             # Skip Fusion entries
#             if classifier == "Fusion":
#                 continue
#             print(details, classifier)
#             # Extract metrics and parameters
#             accuracy = details["test_accuracy"]
#             report = details["classification_report"]
            
#             precision = report["macro avg"]["precision"]
#             recall = report["macro avg"]["recall"]
#             f1 = report["macro avg"]["f1-score"]
#             best_params = str(details["best_params"])
            
#             rows.append([
#                 extractor, 
#                 components, 
#                 classifier,  # Added classifier name (e.g., "KNN", "SVM")
#                 best_params, 
#                 accuracy, 
#                 precision, 
#                 recall, 
#                 f1
#             ])

# # Write to CSV
# with open("classifier_results_with_names.csv", "w", newline="") as f:
#     writer = csv.writer(f)
#     writer.writerow(headers)
#     writer.writerows(rows)


import json
import csv
import os
import sys

JSON_PATH = "results.json"                 # ← nome fixo
CSV_PATH  = "model_classifier_metrics.csv"

HEADERS = [
    "Model", "Classifier", "PCA Components",
    "Best Parameters", "Accuracy",
    "Precision (macro)", "Recall (macro)", "F1-Score (macro)"
]

# Chaves que não representam um classificador “puro”
SKIP = {
    "Fusion", "SoftVoting", "Stacking",
    "disagreement_matrix", "correlation_matrix"
}

# ─── 1. Confere se results.json existe ───────────────────────────────────────
if not os.path.isfile(JSON_PATH):
    sys.exit(f'❌ Arquivo "{JSON_PATH}" não encontrado em {os.getcwd()}.')

# ─── 2. Carrega o JSON ───────────────────────────────────────────────────────
with open("results.json", encoding="utf-8") as f:
    data = json.load(f)

# ─── 3. Extrai as linhas ─────────────────────────────────────────────────────
rows = []
for model, clf_dict in data.items():           # Ex.: ResNet50, EfficientNet…
    for clf_name, details in clf_dict.items(): # Ex.: KNN, SVM…

        if clf_name in SKIP or "test_accuracy" not in details:
            continue

        accuracy = details["test_accuracy"]    # mantido como string

        report      = details["classification_report"]["macro avg"]
        precision   = str(report["precision"])
        recall      = str(report["recall"])
        f1          = str(report["f1-score"])

        best_params = details.get("best_params", {})
        pca_comp    = best_params.get("pca__n_components", "N/A")
        best_params_str = json.dumps(best_params, ensure_ascii=False)

        rows.append([
            model, clf_name, pca_comp,
            best_params_str, accuracy,
            precision, recall, f1
        ])

# Ordena (opcional) por modelo e classificador
rows.sort(key=lambda r: (r[0], r[1]))

# ─── 4. Grava o CSV ───────────────────────────────────────────────────────────
with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
    csv.writer(f).writerows([HEADERS, *rows])

print(f"✅ CSV gerado: {os.path.abspath(CSV_PATH)}")

