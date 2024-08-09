import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def plot_mirror_histogram(data, var, group, bins=50, weights=None, xlabel=None, ylabel=None, title=None):
    """
    Plots a vertically mirrored histogram of a variable by another grouping binary variable.
    
    Parameters:
    -----------
    - data: DataFrame containing the data.
    - var: Name of the column containing the propensity scores.
    - group: Name of the column containing the treatment/control grouping.
    - bins: Number of bins for the histograms (default is 50).
    - xlabel: Label for the x-axis.
    - ylabel: Label for the y-axis.
    - title: Title of the plot.
    """
    # Check if the grouping variable is a binary variable
    unique_groups = data[group].unique()
    if len(unique_groups) != 2:
        raise ValueError("The grouping variable must have exactly two unique values.")
    
    group1, group2 = unique_groups

    if weights:
        weights_group1 = data.query(f"{group} == @group1")[weights]
        weights_group2 = data.query(f"{group} == @group2")[weights]
    else:
        weights_group1 = None
        weights_group2 = None
    
    sns.histplot(
        x = data.query(f"{group} == @group1")[var],
        bins=bins, 
        weights=weights_group1,
        edgecolor='white', color='#0072B2', 
        label=f'Group {group1}'
    )
    
    heights, bins = np.histogram(
        a = data.query(f"{group} == @group2")[var], 
        bins=bins, 
        weights=weights_group2
    )
    heights *= -1  # Reverse the heights for the second group
    bin_width = np.diff(bins)[0]
    bin_pos = bins[:-1] + bin_width / 2
    
    plt.bar(
        bin_pos, heights, width=bin_width, edgecolor='white', 
        color='#D55E00', label=f'Group {group2}'
    )
    
    # Adjust y-axis to show positive values for both groups
    ticks = plt.gca().get_yticks()
    plt.gca().set_yticks(ticks)
    plt.gca().set_yticklabels([abs(int(tick)) for tick in ticks])
    
    if xlabel is None:
        xlabel = f'{var}'
    if ylabel is None:
        ylabel = 'Frequency'
    if title is None:
        title = f'Mirror Histogram of {var} by {group}'
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.show()
    