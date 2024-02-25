
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pdb
import seaborn as sns

def generate_graphs(filepath):
    # Read the csv file
    df = pd.read_csv(filepath)
    
    # Set the seaborn style
    sns.set(style="whitegrid")
    
    # Create a figure with two subplots stacked on top of each other
    fig, axs = plt.subplots(2, 1, figsize=(10, 12))
    
    # First subplot - Linear Scale
    sns.lineplot(ax=axs[0], x=df["N"], y=df["Speedup"], label="Speedup", linewidth=2.5, marker='o', color='blue')
    axs[0].set_xlabel("Matrix Size (N)", fontsize=14)
    axs[0].set_ylabel("X Speedup (GPU/CPU)", fontsize=14)
    axs[0].set_title("Linear Scale: Matrix Size vs Speedup", fontsize=16)
    axs[0].tick_params(labelsize=12)
    axs[0].legend(fontsize=12, title_fontsize='13')
    
    # Second subplot - Logarithmic Scale
    sns.lineplot(ax=axs[1], x=df["N"], y=df["Speedup"], label="Speedup (Log Scale)", linewidth=2.5, marker='o', color='green')
    axs[1].set_xlabel("Matrix Size (N)", fontsize=14)
    axs[1].set_ylabel("Log X Speedup (GPU/CPU)", fontsize=14)
    axs[1].set_title("Logarithmic Scale: Matrix Size vs Speedup", fontsize=16)
    axs[1].set_xscale("log")
    axs[1].set_yscale("log")
    axs[1].tick_params(labelsize=12)
    axs[1].legend(fontsize=12, title_fontsize='13')
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(f"{filepath}_N_vs_Speedup_combined.png")
    


if __name__ == "__main__":
    # Generate graphs for all csv files in this directory
    for file in os.listdir():
        if file.endswith(".csv"):
            generate_graphs(file)