"""
Seaborn visualization utilities for ROUND benchmark suite.
Provides consistent styling and data transformation functions.
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional, Tuple, Union

# Color Schemes
CLASSIC_PALETTE = {
    'ROUND': '#FF4B4B',
    'GRU': '#4B4BFF'
}

ENHANCED_PALETTE = {
    'ROUND': sns.color_palette("rocket")[3],
    'GRU': sns.color_palette("mako")[3]
}

def setup_seaborn_theme(style='darkgrid', palette='classic'):
    """
    Configure Seaborn theme for ROUND benchmarks.

    Args:
        style: 'darkgrid' (default), 'dark', 'whitegrid', 'white'
        palette: 'classic' (original colors) or 'enhanced' (perceptually uniform)

    Returns:
        dict: Color palette to use for plotting
    """
    color_palette = CLASSIC_PALETTE if palette == 'classic' else ENHANCED_PALETTE

    sns.set_theme(
        style=style,
        context='paper',
        rc={
            'figure.facecolor': '#0e1117',
            'axes.facecolor': '#0e1117',
            'grid.alpha': 0.1,
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'font.size': 14,
            'axes.titlesize': 18,
            'axes.labelsize': 16,
            'legend.fontsize': 14,
            'xtick.labelsize': 14,
            'ytick.labelsize': 14,
            'text.color': 'white',
            'axes.labelcolor': 'white',
            'xtick.color': 'white',
            'ytick.color': 'white'
        }
    )
    return color_palette

def prepare_comparison_data(
    round_stats: Union[List[List[float]], np.ndarray],
    gru_stats: Union[List[List[float]], np.ndarray],
    epochs: Optional[np.ndarray] = None
) -> pd.DataFrame:
    """
    Convert ROUND vs GRU statistics to tidy DataFrame.

    Args:
        round_stats: List of runs, each run is list of accuracies
        gru_stats: List of runs, each run is list of accuracies
        epochs: Optional epoch array, defaults to 0..len(run)-1

    Returns:
        DataFrame with columns: ['Epoch', 'Accuracy', 'Model', 'Run']
    """
    # Convert to list if numpy array
    if isinstance(round_stats, np.ndarray):
        round_stats = round_stats.tolist()
    if isinstance(gru_stats, np.ndarray):
        gru_stats = gru_stats.tolist()

    if epochs is None:
        epochs = np.arange(len(round_stats[0]))

    records = []
    for model_name, stats in [('ROUND', round_stats), ('GRU', gru_stats)]:
        for run_idx, run_data in enumerate(stats):
            for epoch_idx, acc in enumerate(run_data):
                records.append({
                    'Epoch': epochs[epoch_idx] if isinstance(epochs, np.ndarray) else epoch_idx,
                    'Accuracy': acc,
                    'Model': model_name,
                    'Run': run_idx
                })
    return pd.DataFrame(records)

def plot_benchmark_comparison(
    df: pd.DataFrame,
    title: str,
    palette: Dict[str, str],
    output_path: str,
    figsize: Tuple[int, int] = (10, 6),
    errorbar: str = 'sd',
    ylabel: str = 'Accuracy',
    xlabel: str = 'Epochs'
) -> None:
    """
    Create standard ROUND vs GRU comparison plot.

    Args:
        df: DataFrame from prepare_comparison_data()
        title: Plot title
        palette: Color dictionary {'ROUND': color, 'GRU': color}
        output_path: Full path for saving figure
        figsize: Figure dimensions
        errorbar: 'sd' for std dev, ('ci', 95) for 95% CI
        ylabel: Y-axis label
        xlabel: X-axis label
    """
    fig, ax = plt.subplots(figsize=figsize)

    sns.lineplot(
        data=df,
        x='Epoch',
        y='Accuracy',
        hue='Model',
        palette=palette,
        linewidth=2.5,
        errorbar=errorbar,
        err_style='band',
        err_kws={'alpha': 0.15},
        ax=ax
    )

    ax.set_title(title, fontsize=18, color='white', weight='bold')
    ax.set_xlabel(xlabel, fontsize=16, color='white')
    ax.set_ylabel(ylabel, fontsize=16, color='white')
    ax.legend(loc='best', frameon=True, fancybox=True, fontsize=14)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_benchmark_with_runs(
    df: pd.DataFrame,
    title: str,
    palette: Dict[str, str],
    output_path: str,
    figsize: Tuple[int, int] = (12, 6),
    errorbar: str = 'sd',
    ylabel: str = 'Accuracy',
    xlabel: str = 'Epochs',
    ylim: Optional[Tuple[float, float]] = None
) -> None:
    """
    Create ROUND vs GRU comparison plot showing individual runs as faint lines.
    Used by benchmark_topology.py.

    Args:
        df: DataFrame from prepare_comparison_data()
        title: Plot title
        palette: Color dictionary {'ROUND': color, 'GRU': color}
        output_path: Full path for saving figure
        figsize: Figure dimensions
        errorbar: 'sd' for std dev, ('ci', 95) for 95% CI
        ylabel: Y-axis label
        xlabel: X-axis label
        ylim: Optional y-axis limits as (ymin, ymax)
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Plot individual runs (faint)
    sns.lineplot(
        data=df,
        x='Epoch',
        y='Accuracy',
        hue='Model',
        units='Run',
        estimator=None,
        alpha=0.15,
        linewidth=1,
        palette=palette,
        legend=False,
        ax=ax
    )

    # Overlay mean with error bands
    sns.lineplot(
        data=df,
        x='Epoch',
        y='Accuracy',
        hue='Model',
        linewidth=2.5,
        errorbar=errorbar,
        err_kws={'alpha': 0.15},
        palette=palette,
        ax=ax
    )

    ax.set_title(title, fontsize=18, color='white', weight='bold')
    ax.set_xlabel(xlabel, fontsize=16, color='white')
    ax.set_ylabel(ylabel, fontsize=16, color='white')

    if ylim:
        ax.set_ylim(ylim)

    ax.legend(loc='lower right', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def prepare_multi_word_data(
    hist_dict: Dict[str, List[float]],
    words: List[str],
    ep_axis: np.ndarray,
    model_name: str
) -> pd.DataFrame:
    """
    Convert word-by-word history to tidy DataFrame.
    Used by benchmark_phase_lock.py.

    Args:
        hist_dict: Dictionary mapping word -> list of accuracies
        words: List of words
        ep_axis: Epoch axis array
        model_name: 'ROUND' or 'GRU'

    Returns:
        DataFrame with columns: ['Epoch', 'Accuracy', 'Word', 'Model']
    """
    records = []
    for word in words:
        for epoch_idx, acc in enumerate(hist_dict[word]):
            records.append({
                'Epoch': ep_axis[epoch_idx],
                'Accuracy': acc,
                'Word': word,
                'Model': model_name
            })
    return pd.DataFrame(records)

def plot_multi_word_comparison(
    hist_r: Dict[str, List[float]],
    hist_g: Dict[str, List[float]],
    words: List[str],
    ep_axis: np.ndarray,
    hidden_size_r: int,
    hidden_size_g: int,
    output_path: str,
    word_colors: List[str]
) -> None:
    """
    Create 2-panel plot for word-by-word learning curves.
    Used by benchmark_phase_lock.py.

    Args:
        hist_r: ROUND history dict (word -> accuracies)
        hist_g: GRU history dict (word -> accuracies)
        words: List of words
        ep_axis: Epoch axis array
        hidden_size_r: ROUND hidden size
        hidden_size_g: GRU hidden size
        output_path: Full path for saving figure
        word_colors: List of colors for each word
    """
    fig, (ax_r, ax_g) = plt.subplots(2, 1, figsize=(14, 12))

    # Prepare DataFrames
    df_round = prepare_multi_word_data(hist_r, words, ep_axis, 'ROUND')
    df_gru = prepare_multi_word_data(hist_g, words, ep_axis, 'GRU')

    # Create word palette
    word_palette = {word: word_colors[i % len(word_colors)] for i, word in enumerate(words)}

    # Plot ROUND panel
    sns.lineplot(
        data=df_round,
        x='Epoch',
        y='Accuracy',
        hue='Word',
        palette=word_palette,
        linewidth=2,
        ax=ax_r
    )
    ax_r.set_title(f"ROUND - Phase Angle Lock ({hidden_size_r} Neurons)",
                   color='#FF5555', fontsize=20, weight='bold')
    ax_r.legend(loc='lower left', fontsize=12, ncol=3)
    ax_r.set_xlabel('Epochs', fontsize=16, color='white')
    ax_r.set_ylabel('Accuracy', fontsize=16, color='white')

    # Plot GRU panel
    sns.lineplot(
        data=df_gru,
        x='Epoch',
        y='Accuracy',
        hue='Word',
        palette=word_palette,
        linewidth=2,
        linestyle='--',
        ax=ax_g
    )
    ax_g.set_title(f"GRU - Standard Gating ({hidden_size_g} Neurons)",
                   color='#5555FF', fontsize=20, weight='bold')
    ax_g.legend(loc='lower left', fontsize=12, ncol=3)
    ax_g.set_xlabel('Epochs', fontsize=16, color='white')
    ax_g.set_ylabel('Accuracy', fontsize=16, color='white')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
