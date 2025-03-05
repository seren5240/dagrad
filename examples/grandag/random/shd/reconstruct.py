import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

num_nodes = [5, 10, 20]
s0_ratios = [1, 2, 4]
noise_types = ["gauss", "exp", "gumbel"]
methods = ["GRAN-DAG", "NOTEARS"]

noise_names = {
    "gauss": "Gaussian",
    "exp": "Exponential",
    "gumbel": "Gumbel"
}

method_labels = {
    "GRAN-DAG": "GRAN-DAG with CAM pruning",
    "NOTEARS": "NOTEARS"
}

num_rows = len(s0_ratios)
num_cols = len(noise_types)
fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows), sharex=True, sharey=True)

file_pattern = "grandag_shd_ER*_noise=*_n=1000_var=random.txt"
files = sorted(glob.glob(file_pattern))

results = {method: {sem: {str(s0): {d: [] for d in num_nodes} for s0 in s0_ratios} for sem in noise_types} for method in methods}

for file in files:
    base_name = os.path.basename(file)
    parts = base_name.replace("grandag_", "").replace("_n=1000_var=random.txt", "").split("_")

    er_type = parts[1].replace("ER", "")
    noise_type = parts[2].split("=")[1]

    # if er_type != '1':
    #     continue

    df = pd.read_csv(file)

    for method in methods:
        sub_df = df[df['method'] == method]
        for _, row in sub_df.iterrows():
            d = row["d"]
            results[method][noise_type][er_type][d].append(row["mean_normalized_shd"])

print(f'results are {results}')

for i, s0_ratio in enumerate(s0_ratios):
    for j, noise in enumerate(noise_types):
        ax = axes[i, j] if num_rows > 1 else axes[j]

        for method in methods:
            means = [np.mean(results[method][noise][str(s0_ratio)][d]) for d in num_nodes]
            ax.plot(num_nodes, means, marker="o", label=method_labels[method])

        ax.set_title(f"{noise_names[noise]} noise, ER{s0_ratio}")
        ax.set_xlabel("d (Number of Nodes)")
        if j == 0:
            ax.set_ylabel("Normalized SHD")
        ax.grid(True)

handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=len(methods))
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(f"normalized_shd_n=1000_var=random_trials=10.png")
