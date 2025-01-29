import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

fig, axes = plt.subplots(3, 3, figsize=(12, 12))
fig.subplots_adjust(hspace=0.3, wspace=0.3)

file_pattern = "golem_ER*_noise=*_n=1000_var=random.txt"

files = glob.glob(file_pattern)

files.sort()

row_labels = ["ER1", "ER2", "ER4"]
col_labels = ["Gaussian noise", "Exponential noise", "Gumbel noise"]

noise_names = {
    'gauss': 'Gaussian',
    'gumbel': 'Gumbel',
    'exp': 'Exponential',
}

for file in files:
    base_name = os.path.basename(file)
    parts = base_name.replace("golem_", "").replace("_n=1000_var=random.txt", "").split("_")
    
    er_type = parts[0]
    noise_type = parts[1].split("=")[1]
    
    row = row_labels.index(er_type)
    col = col_labels.index(f"{noise_names[noise_type]} noise")
    
    ax = axes[row, col]
    
    df = pd.read_csv(file)
    
    for method in df['method'].unique():
        sub_df = df[df['method'] == method]
        ax.plot(sub_df['d'], sub_df['mean_normalized_shd'], marker='o', label=method)
    
    ax.set_title(f"{noise_type} noise, {er_type}")
    ax.set_xlabel("d (Number of Nodes)")
    ax.set_ylabel("Normalized SHD")

handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=2)

plt.show()
