
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pdb
import seaborn as sns

def generate_graphs(filepath, results_dir="results"):
    # extract filename from filepath
    filename = filepath.split("/")[-1]

    # Read the csv file
    df = pd.read_csv(filepath)
    
    # Set the seaborn style
    sns.set_theme(style="whitegrid")
    
    # Create a figure with two subplots stacked on top of each other
    fig, axs = plt.subplots(2, 1, figsize=(10, 12))
    
    # First subplot - Linear Scale 
    sns.lineplot(ax=axs[0], x=df["N"], y=df["Speedup"], label="Speedup", linewidth=2.5, marker='o', color=np.random.rand(3,))
    axs[0].set_xlabel("Size (N)", fontsize=14)
    axs[0].set_ylabel("X Speedup (GPU/CPU)", fontsize=14)
    axs[0].set_title(f"{filename}: Size vs Speedup", fontsize=16)
    axs[0].tick_params(labelsize=12)
    axs[0].legend(fontsize=12, title_fontsize='13')
    
    # Second subplot - Logarithmic Scale
    sns.lineplot(ax=axs[1], x=df["N"], y=df["Speedup"], label="Speedup (Log Scale)", linewidth=2.5, marker='o', color=np.random.rand(3,))
    axs[1].set_xlabel("Matrix Size (N)", fontsize=14)
    axs[1].set_ylabel("Log X Speedup (GPU/CPU)", fontsize=14)
    axs[1].set_title(f"{filename}: Size vs Speedup (Log scale)", fontsize=16)
    axs[1].set_xscale("log")
    axs[1].set_yscale("log")
    axs[1].tick_params(labelsize=12)
    axs[1].legend(fontsize=12, title_fontsize='13')
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    # Save the figure in results directory
    plt.savefig(f"{results_dir}/{filename}_N_vs_Speedup.png")
    


if __name__ == "__main__":
    # Generate graphs for all csv files in this directory
    # files are in ..data/
    for file in os.listdir("data/"):
        if file.endswith(".csv"):
            generate_graphs("data/" + file)