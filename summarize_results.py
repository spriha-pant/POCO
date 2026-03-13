import os
import numpy as np
from collections import defaultdict

# BASE_DIR = "/home/arnablab/POCO/experiments/zapbench_seed5"
BASE_DIR = "/home/arnablab/POCO/experiments/barikmousemousmi_seed5"

results = []

for root, dirs, files in os.walk(BASE_DIR):
    if "test.txt" in files:
        folder_name = os.path.basename(root)

        try:
            # Extract dataset
            dataset = folder_name.split("dataset_label")[1].split("_model_label")[0]

            # Extract model
            model = folder_name.split("model_label")[1].split("_s")[0]

            # Extract seed
            seed = int(folder_name.split("_s")[-1])
        except Exception:
            continue

        test_path = os.path.join(root, "test.txt")

        with open(test_path, "r") as f:
            lines = f.readlines()

        if len(lines) < 2:
            continue

        headers = lines[0].strip().split("\t")
        values = lines[1].strip().split("\t")

        # if "celegans_val_score" in headers:
        #     score_idx = headers.index("celegans_val_score")
        # elif "celegansflavell_val_score" in headers:
        #     score_idx = headers.index("celegansflavell_val_score")
        # elif "zebrafishahrens_val_score" in headers:
        #     score_idx = headers.index("zebrafishahrens_val_score")
        if "barikmousemousmi_val_score" in headers:
            score_idx = headers.index("barikmousemousmi_val_score")
        else:
            # generic fallback: find any column ending with _val_score
            score_idx = None
            for i, h in enumerate(headers):
                if h.endswith("_val_score"):
                    score_idx = i
                    break

        if score_idx is not None:
            score = float(values[score_idx])
            results.append((dataset, model, seed, score))


# ---------- PRINT INDIVIDUAL RUNS ----------

print("\nINDIVIDUAL RUNS")
print("=" * 85)
print(f"{'Dataset':<25} {'Model':<10} {'Seed':<5} {'Score':<10}")
print("-" * 85)

for dataset, model, seed, score in sorted(results):
    print(f"{dataset:<25} {model:<10} {seed:<5} {score:<10.4f}")


# ---------- COMPUTE MEAN ± STD ----------

grouped = defaultdict(list)

for dataset, model, seed, score in results:
    grouped[(dataset, model)].append(score)

print("\n\nMEAN ± STD ACROSS SEEDS")
print("=" * 85)
print(f"{'Dataset':<25} {'Model':<10} {'Mean':<10} {'Std':<10}")
print("-" * 85)

for (dataset, model), scores in sorted(grouped.items()):
    mean = np.mean(scores)
    std = np.std(scores)
    print(f"{dataset:<25} {model:<10} {mean:<10.4f} {std:<10.4f}")