# utils/logger.py
import json

def save_results(results, filename="results.json"):
    with open(filename, "w") as f:
        json.dump(results, f, indent=4)