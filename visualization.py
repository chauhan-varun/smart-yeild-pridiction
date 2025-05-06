"""
Visualization functions for agricultural yield prediction.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
import dask.dataframe as dd


def create_output_dir():
    """Create output directory for visualizations if it doesn't exist."""
    os.makedirs('visualizations', exist_ok=True)


def plot_yield_distribution(df):
    """
    Plot the distribution of crop yields.
    
    Args:
        df (dask.dataframe.DataFrame): The dataset
    """
    print("Plotting yield distribution...")
    create_output_dir()
    
    # Sample the data to make visualization faster
    df_sample = df['Yield'].compute().sample(min(10000, len(df)))
    
    plt.figure(figsize=(10, 6))
    sns.histplot(df_sample, kde=True)
    plt.title('Distribution of Crop Yields')
    plt.xlabel('Yield')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig('visualizations/yield_distribution.png')
    plt.close()


def plot_yield_by_crop(df):
    """
    Plot average yield by crop type.
    
    Args:
        df (dask.dataframe.DataFrame): The dataset
    """
    print("Plotting yield by crop...")
    create_output_dir()
    
    # Compute average yield by crop
    crop_yield = df.groupby('Crop')['Yield'].mean().compute().sort_values(ascending=False)
    top_crops = crop_yield.head(15)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x=top_crops.values, y=top_crops.index)
    plt.title('Average Yield by Crop Type (Top 15)')
    plt.xlabel('Average Yield')
    plt.ylabel('Crop')
    plt.tight_layout()
    plt.savefig('visualizations/yield_by_crop.png')
    plt.close()


def plot_yield_by_season(df):
    """
    Plot average yield by season.
    
    Args:
        df (dask.dataframe.DataFrame): The dataset
    """
    print("Plotting yield by season...")
    create_output_dir()
    
    # Compute average yield by season
    season_yield = df.groupby('Season')['Yield'].mean().compute()
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=season_yield.index, y=season_yield.values)
    plt.title('Average Yield by Season')
    plt.xlabel('Season')
    plt.ylabel('Average Yield')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('visualizations/yield_by_season.png')
    plt.close()


def plot_yield_by_year(df):
    """
    Plot average yield by year.
    
    Args:
        df (dask.dataframe.DataFrame): The dataset
    """
    print("Plotting yield by year...")
    create_output_dir()
    
    # Compute average yield by year
    year_yield = df.groupby('Crop_Year')['Yield'].mean().compute()
    
    plt.figure(figsize=(12, 6))
    sns.lineplot(x=year_yield.index, y=year_yield.values, marker='o')
    plt.title('Average Yield by Year')
    plt.xlabel('Year')
    plt.ylabel('Average Yield')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('visualizations/yield_by_year.png')
    plt.close()


def plot_correlation_matrix(df):
    """
    Plot correlation matrix for numeric features.
    
    Args:
        df (dask.dataframe.DataFrame): The dataset
    """
    print("Plotting correlation matrix...")
    create_output_dir()
    
    # Select numeric columns and convert to pandas
    numeric_cols = ['Area', 'Production', 'Yield', 'Crop_Year']
    df_sample = df[numeric_cols].sample(frac=0.1).compute()
    
    # Calculate correlation matrix
    corr_df = df_sample.corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_df, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Matrix of Numeric Features')
    plt.tight_layout()
    plt.savefig('visualizations/correlation_matrix.png')
    plt.close()


def plot_yield_by_state(df):
    """
    Plot average yield by state.
    
    Args:
        df (dask.dataframe.DataFrame): The dataset
    """
    print("Plotting yield by state...")
    create_output_dir()
    
    # Compute average yield by state
    state_yield = df.groupby('State')['Yield'].mean().compute().sort_values(ascending=False)
    top_states = state_yield.head(15)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x=top_states.values, y=top_states.index)
    plt.title('Average Yield by State (Top 15)')
    plt.xlabel('Average Yield')
    plt.ylabel('State')
    plt.tight_layout()
    plt.savefig('visualizations/yield_by_state.png')
    plt.close()


def create_visualizations(df):
    """
    Create all visualizations for the dataset.
    
    Args:
        df (dask.dataframe.DataFrame): The dataset
    """
    print("Creating visualizations...")
    
    # Create individual visualizations
    plot_yield_distribution(df)
    plot_yield_by_crop(df)
    plot_yield_by_season(df)
    plot_yield_by_year(df)
    plot_correlation_matrix(df)
    plot_yield_by_state(df)
    
    print("Visualizations created and saved to 'visualizations' directory")
