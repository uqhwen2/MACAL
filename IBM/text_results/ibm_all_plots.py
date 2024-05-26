#%%
import glob
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
# Path to the directory containing your CSV files
# files_path = "two_stage/*.csv"
labels = [
          'mu',
          'rho',
          'murho',
          'qhte',
          'random',
          'lcmd',
          'kmeanspp',
          'bait',
          'macal_0.0',
          'macal_1.0',
          'macal_2.5',
          'macal_5.0',
          'macal_10.0',
]

# Define a color gradient for shading
colors = ['lightblue', 'blue']  # Define your colors
cmap = LinearSegmentedColormap.from_list("my_cmap", colors)

# colors is a list of colors you want to use for each plot
colours = [
           'blue',
           'orange',
           'green', 
           'black', 
           'purple',
           'brown',
           'pink', 
           'gray', 
           'olive', 
           'cyan', 
           'red', 
           'lime', 
           'magenta'
]


for index, files_path in enumerate([
                                    "truemu/*.csv",
                                    "truerho/*.csv", 
                                    "truemurho/*.csv", 
                                    "trueqhte/*.csv",
                                    'truerandom/*.csv',
                                    "truelcmd/*.csv", 
                                    "truekmeanspp/*.csv",
                                    "truebait/*.csv",
                                    "truesim_0.0/*.csv",
                                    "truesim_1.0/*.csv",
                                    "truesim_2.5/*.csv",
                                    "truesim_5.0/*.csv",
                                    "truesim_10.0/*.csv",
]):
    
    # List to store dataframes for each file
    dfs = []

    try:
        # Read each CSV file and append it to the list
        for file in glob.glob(files_path):
            df = pd.read_csv(file, index_col=0)
            dfs.append(df.loc["PEHE"])  # Extract the row corresponding to "PEHE" for each file
    except pd.errors.EmptyDataError:
        # Skip this iteration if there are no files in the directory
        continue

    # Check if there are no files in the directory
    if not dfs:
        continue

    # Concatenate all dataframes into a single dataframe
    combined_df = pd.concat(dfs, axis=1)
    
    # Calculate the mean for each "number of samples"
    mean_values = combined_df.mean(axis=1)
    
    # Calculate the variance for each "number of samples"
    std_values = combined_df.std(axis=1)
    
    # Assuming you have a Pandas Series for mean and variance
    mean_series = mean_values[:50]
    std_series = std_values[:50]
    
    # Plot mean using .index and .values
    plt.plot(mean_series.index, mean_series.values, marker='o', markersize=5, linestyle='-', label=f"{labels[index]}", color=colours[index])
    
    # Plot variance using .index and .values, and fill between +/- one standard deviation
#    plt.fill_between(std_series.index, 
#                     mean_series - std_series, 
#                     mean_series + std_series,
#                     cmap=cmap,  # Use the defined colormap
#                     alpha=0.3)  # Adjust transparency as needed     

# Add labels and title
plt.xlim(0)
plt.xlabel("Training Samples")
plt.ylabel("PEHE")

# Adjust legend to display horizontally and set the position
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=9)

# Tilt x-axis labels
plt.xticks(rotation=25, ticks=list(range(0, 50, 3)), labels=list(range(50, 2501, 150)))  # Adjust the rotation angle as needed

# Save the figure
plt.savefig('IBM-PEHE.pdf', bbox_inches='tight')  # Change the filename and extension as needed

# Show the plot
plt.show()
#%%
