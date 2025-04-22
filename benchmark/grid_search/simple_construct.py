import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

num_nodes = [5, 10, 20, 50, 100]
s0_ratios = [1, 2, 4]
noise_types = ["gauss", "exp", "gumbel"]

noise_names = {"gauss": "Gaussian", "exp": "Exponential", "gumbel": "Gumbel"}
error_vars = ["eq", "random"]

pattern = re.compile(r"notears_linear_lmd=(.*?)_gamma=(.*?)_rho_init=(.*?)_mcp_loss\.txt")

param_set = ['Base NOTEARS']

for filename in os.listdir("."):
    match = pattern.match(filename)
    if match:
        lmd, gamma, rho_init = match.groups()
        param_set.append(f"lmd={lmd}, gamma={gamma}, rho_init={rho_init}")

results = {
    ps: {
        ev: {
            sem: {str(s0): {d: [] for d in num_nodes} for s0 in s0_ratios}
            for sem in noise_types
        }
        for ev in error_vars
    }
    for ps in param_set
}

for filename in os.listdir("."):
    match = pattern.match(filename)
    if match:
        lmd, gamma, rho_init = match.groups()
        key = f"lmd={lmd}, gamma={gamma}, rho_init={rho_init}"
        df = pd.read_csv(filename)
    elif filename == 'notears_basic.txt':
        key = 'Base NOTEARS'
        df = pd.read_csv(filename)
    else:
        continue

    for _, row in df.iterrows():
        d = row["d"]
        edges = int(row["edges"])
        error_var = row["error_var"]
        noise_type = row["noise_type"]
        er_type = str(int(edges / d))
        results[key][error_var][noise_type][er_type][d].append(row["mean_normalized_shd"])

top_configs_by_error_var = {}

for ev in error_vars:
    config_means = {}
    for ps in param_set:
        if ps == "Base NOTEARS":
            continue
        values = []
        for noise in noise_types:
            for s0 in s0_ratios:
                for d in num_nodes:
                    values.extend(results[ps][ev][noise][str(s0)][d])
        config_means[ps] = np.mean(values) if values else float("inf")
    
    top5 = sorted(config_means, key=config_means.get)[:5]
    top_configs_by_error_var[ev] = ["Base NOTEARS"] + top5

def make_error_var_plot(error_var):
    top_configs = top_configs_by_error_var[error_var]

    num_rows = len(s0_ratios)
    num_cols = len(noise_types)
    fig, axes = plt.subplots(
        num_rows, num_cols, figsize=(15, 5 * num_rows), sharex=True, sharey=True
    )

    for i, s0_ratio in enumerate(s0_ratios):
        for j, noise in enumerate(noise_types):
            ax = axes[i, j] if num_rows > 1 else axes[j]

            for ps in param_set:
                means = [
                    np.mean(results[ps][error_var][noise][str(s0_ratio)][d])
                    for d in num_nodes
                ]
                color = None if ps in top_configs else "grey"
                alpha = 1.0 if ps in top_configs else 0.3
                lw = 2 if ps in top_configs else 1
                ax.plot(
                    num_nodes, means, marker="o", label=ps,
                    color=color, alpha=alpha, linewidth=lw
                )

            ax.set_title(f"{noise_names[noise]} noise, ER{s0_ratio}")
            ax.set_xlabel("d (Number of Nodes)")
            if j == 0:
                ax.set_ylabel("Normalized SHD")
            ax.grid(True)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    top_handles = [h for h, l in zip(handles, labels) if l in top_configs]
    top_labels = [l for l in labels if l in top_configs]

    fig.legend(
        top_handles, top_labels,
        loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0), fontsize="small"
    )
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.suptitle(f"Linear SEM, var={error_var}", y=0.97)
    plt.savefig(f"normalized_shd_n=1000_var={error_var}_top5.png")


make_error_var_plot("eq")
make_error_var_plot("random")
