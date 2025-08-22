import os
import matplotlib.pyplot as plt

datasets = {
    #"cleaned": "games_march2025_cleaned",
    "cleaned_2k": "games_march2025_cleaned_2k_i3k",
    "cleaned_10k": "games_march2025_cleaned_10k_i3k"
}
# def results
results = {}

for dataset_name, folder in datasets.items():
    results[dataset_name] = {}
    for filename in os.listdir(folder):
        if filename.endswith(".txt"):
            model_name = filename.replace(".txt", "")
            with open(os.path.join(folder, filename), "r") as f:
                for line in f:
                    if line.strip().startswith("weighted avg"):
                        parts = line.split()
                        f1_score = float(parts[4])  # precision recall f1-score support
                        results[dataset_name][model_name] = f1_score

# Plot
#models = sorted(results["cleaned_2k"].keys())  # alphabetisch sortieren für gleiche Reihenfolge
models = dict(sorted(results["cleaned_2k"].items(), key=lambda i: i[1], reverse=True)) # nach values sortieren
x = range(len(models))

plt.figure(figsize=(12,6))
#plt.bar([i - 0.25 for i in x], [results["cleaned"][m] for m in models], width=0.25, label="cleaned")
plt.bar(x, [results["cleaned_2k"][m] for m in models], width=0.25, label="2k Dataset")
plt.bar([i + 0.25 for i in x], [results["cleaned_10k"].get(m,0) for m in models], width=0.25, label="10k Dataset")

plt.xticks(x, models, rotation=90)
plt.ylim(0, 1) # min max
plt.ylabel("Weighted F1-Score")
plt.title("Model Performance across Datasets")
plt.legend()
plt.tight_layout()
plt.savefig('compare_graph_latest.png')
plt.show()
