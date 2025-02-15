import matplotlib.pyplot as plt
import numpy as np

# Data for the models and datasets
models = ['DKT', 'SAKT']  # Removed 'AKT'
datasets = ['statics2011', 'Assistments2009', 'Assistments2017', 'kddcup2010']
scores = {
    'DKT': [0.8106, 0.8023, 0.7052, 0.7874],
    'DKT with DGSPM': [0.8209, 0.8102, 0.7183, 0.7960],
    'SAKT': [0.8022, 0.7361, 0.6492, 0.7736],
    'SAKT with DGSPM': [0.8108, 0.7453, 0.6561, 0.7852],
}

# Adjusting the y-axis range slightly higher to avoid legend overlapping with the value labels
fig, axes = plt.subplots(1, 2, figsize=(30, 10))  # Adjusted to 2 subplots
width = 0.40  # Bar width

# Adjust y-axis limits to be slightly higher
y_limits_adjusted = {
    'DKT': (0.70, 0.86),
    'SAKT': (0.64, 0.88),
}

# Set font family to 'Arial' (or any sans-serif font for black body)
plt.rcParams['font.family'] = 'Arial'  # Or 'sans-serif'

for idx, model in enumerate(models):
    base_model = scores[model]
    hd_model = scores[f"{model} with DGSPM"]
    x = np.arange(len(datasets))

    # Plotting bars with edgecolor and linewidth settings, and specifying colors
    bars_base = axes[idx].bar(x - width / 2, base_model, width, label=f"{model}", alpha=0.7, edgecolor='black', linewidth=2, color='#D6EFD8')
    bars_hd = axes[idx].bar(x + width / 2, hd_model, width, label=f"{model} with DGSPM", alpha=0.7, edgecolor='black', linewidth=2, color='#FFB0B0')

    # Adding labels on top of the bars
    for bar in bars_base:
        height = bar.get_height()
        axes[idx].text(bar.get_x() + bar.get_width() / 2, height + 0.001, f'{height:.3f}',  # Rounded to 3 decimal places
                       ha='center', va='bottom', fontsize=20)
    for bar in bars_hd:
        height = bar.get_height()
        axes[idx].text(bar.get_x() + bar.get_width() / 2, height + 0.001, f'{height:.3f}',  # Rounded to 3 decimal places
                       ha='center', va='bottom', fontsize=20)

    # Set titles and axis labels with larger font sizes
    axes[idx].set_title(model, fontsize=32)
    axes[idx].set_xticks(x)
    axes[idx].set_xticklabels(datasets, fontsize=18, rotation=45)
    axes[idx].set_xlabel("Datasets", fontsize=28)
    axes[idx].set_ylabel("AUC", fontsize=28)
    axes[idx].spines['left'].set_linewidth(3)
    axes[idx].spines['bottom'].set_linewidth(3)
    axes[idx].set_ylim(y_limits_adjusted[model])  # Adjusted y-axis limits
    axes[idx].legend(fontsize=26, framealpha=0.0)
    axes[idx].grid(axis='y', linestyle='--', alpha=0.7, linewidth=2)
    axes[idx].tick_params(axis='y', labelsize=26,)
    axes[idx].tick_params(axis='x', labelsize=26,)
    for label in axes[idx].get_xticklabels():
        label.set_rotation(0)  # 将x轴标签旋转45度

fig.suptitle("", fontsize=22)
plt.tight_layout()
plt.show()
