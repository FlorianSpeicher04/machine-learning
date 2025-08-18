import os
import matplotlib.pyplot as plt
import numpy as np

datasets = {
    #"cleaned": "games_march2025_cleaned",
    #"cleaned_2k": "games_march2025_cleaned_2k",
    #"cleaned_10k": "games_march2025_cleaned_10k"
    "cleaned_2k": "games_march2025_cleaned_2k_i3k",
}
# def results
results = {}

for dataset_name, folder in datasets.items():
    results[dataset_name] = {}
    for filename in os.listdir(folder):
        if filename.endswith(".txt"):
            model_name = filename.replace(".txt", "")
            print("model " + model_name)
            results[dataset_name][model_name] = {}
            with open(os.path.join(folder, filename), "r") as f:
                for line in f:
                    if line.strip().startswith("micro avg"):
                        print("micro")
                        results[dataset_name][model_name][0] = float(line.split()[4]) # micro f1
                    if line.strip().startswith("macro avg"):
                        print("macro")
                        results[dataset_name][model_name][1] = float(line.split()[4]) # macro f1
                    if line.strip().startswith("weighted avg"):
                        print("weight")
                        results[dataset_name][model_name][2] = float(line.split()[4]) # weighted avg f1

# Plot
#models = sorted(results["cleaned_2k"].keys())  # alphabetisch sortieren für gleiche Reihenfolge
models = dict(sorted(results["cleaned_2k"].items(), key=lambda i: i[1][2], reverse=True)) # nach values sortieren
print(models)
x = range(len(models))

fig = plt.figure()
#ax = fig.add_subplot(projection='3d')

plt.bar([i - 0.25 for i in x], [results["cleaned_2k"][m][0] for m in models], width=0.25, label="Micro")
plt.bar(x,                     [results["cleaned_2k"][m][1] for m in models], width=0.25, label="Macro")
plt.bar([i + 0.25 for i in x], [results["cleaned_2k"][m][2] for m in models], width=0.25, label="Weighted")

plt.xticks(x, models, rotation=90)
plt.ylabel("F1 Score")
#ax.set_zlabel("F1 Value")
plt.ylim(0,1)
plt.title("Model Performance - 2k Dataset")
plt.legend()
plt.tight_layout()
plt.savefig('compare_graph_latest_3.png')
plt.show()

# On the y-axis let's only label the discrete values that we have data for.
#ax.set_yticks(yticks)

plt.show()