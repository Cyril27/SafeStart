import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

df = pd.read_csv("../skeleton_metrics.csv")

#Compare skeleton metrics when using method 1 or 3
def plot_comparison(column):
    plt.figure()
    for i, row in df.iterrows():
        plt.plot(
            [1, 2],
            [row[f'{column}_1'], row[f'{column}_3']],
            marker='o',
            label=f'Row {i}'
        )

    plt.xticks([1, 2], ['Method 1', 'Method 3'])
    plt.xlabel("Method")
    plt.ylabel(f"{column}")
    plt.title(f"Evolution of {column} Across Methods")
    plt.grid(True)
    plt.tight_layout()


plot_comparison('approx_branches')
# plot_comparison('total_length')
# plot_comparison('endpoints')
# plot_comparison('branchpoints')
# plot_comparison('avg_branch_length')
plt.show()