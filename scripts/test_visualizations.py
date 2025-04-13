#!/usr/bin/env python3
"""
Test script for NFL injury visualizations.
This script loads the processed data and displays visualizations directly without saving them.
"""

import sys
from pathlib import Path
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import gc  # Add garbage collection import

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.visualization.plotter import NFLPlotter
from src.utils.logging import setup_logging, logger

def format_statistics(df: pd.DataFrame, title: str) -> None:
    """Format and log statistics in a readable way."""
    logger.info(f"\n{'='*80}")
    logger.info(f"{title:^80}")
    logger.info(f"{'='*80}")
    
    # Format the DataFrame for display
    formatted_df = df.copy()
    
    # Format numbers with commas and 2 decimal places
    for col in ['play_count', 'injury_count']:
        if col in formatted_df.columns:
            formatted_df[col] = formatted_df[col].apply(lambda x: f"{x:,.0f}")
    
    for col in ['injury_rate_quarter', 'injury_rate_down', 'injury_rate_score', 'injury_rate_time']:
        if col in formatted_df.columns:
            formatted_df[col] = formatted_df[col].apply(lambda x: f"{x:,.2f}")
    
    # Log each row with proper spacing
    for _, row in formatted_df.iterrows():
        logger.info("")
        for col in formatted_df.columns:
            if col in ['score_bin', 'time_bin']:
                # Handle interval formatting
                interval = row[col]
                logger.info(f"{col:>20}: {str(interval):<30}")
            elif col in ['score_differential', 'game_seconds_remaining']:
                # Handle interval mid points
                if isinstance(row[col], pd.Interval):
                    logger.info(f"{col:>20}: {row[col].mid:<15.2f}")
                else:
                    logger.info(f"{col:>20}: {row[col]:<15}")
            elif col in ['quarter', 'down']:
                logger.info(f"{col:>20}: {row[col]:<15}")
            else:
                logger.info(f"{col:>20}: {row[col]:<15}")
        logger.info("-"*80)

def convert_score_bin(bin_str):
    """Convert score differential bin string to numeric value."""
    if pd.isna(bin_str) or bin_str == '' or bin_str == 'nan':
        return np.nan
    try:
        # Handle the format "-50--20" or "20-50"
        parts = bin_str.split('-')
        if len(parts) == 3:  # Handle negative numbers like "-50--20"
            lower = float(f"-{parts[1]}")
            upper = float(f"-{parts[2]}")
        else:  # Handle regular ranges like "20-50"
            lower = float(parts[0])
            upper = float(parts[1])
        return (lower + upper) / 2
    except (ValueError, IndexError):
        return np.nan

def main():
    """Main function to test visualizations."""
    try:
        # Setup logging
        setup_logging()
        
        # Set interactive mode
        plt.ion()
        
        # Print matplotlib backend information
        logger.info(f"Using matplotlib backend: {matplotlib.get_backend()}")
        
        # Load processed data
        plays_path = project_root / 'data/processed/processed_plays.parquet'
        injuries_path = project_root / 'data/processed/processed_injuries.parquet'
        
        # Load only necessary columns to reduce memory usage
        plays_cols = ['season', 'week', 'quarter', 'down', 'score_differential', 'game_seconds_remaining', 'play_id']
        
        logger.info(f"Loading data from {plays_path} and {injuries_path}")
        plays_data = pd.read_parquet(plays_path, columns=plays_cols)
        
        # First load injuries data without column filtering to inspect available columns
        injuries_data = pd.read_parquet(injuries_path)
        logger.info(f"Available columns in injuries data: {injuries_data.columns.tolist()}")
        
        # Now filter to only the columns we need
        injuries_cols = ['season', 'week']  # We'll add more columns after inspecting what's available
        injuries_data = injuries_data[injuries_cols]
        
        logger.info(f"Loaded {len(plays_data)} plays and {len(injuries_data)} injuries")
        
        # Create unique game identifiers
        plays_data['game_id'] = plays_data['season'].astype(int).astype(str) + '_' + plays_data['week'].astype(int).astype(str)
        injuries_data['game_id'] = injuries_data['season'].astype(int).astype(str) + '_' + injuries_data['week'].astype(int).astype(str)
        
        # Create plotter instance
        plotter = NFLPlotter(project_root / 'reports/figures')
        
        # Test each visualization
        logger.info("Testing visualizations...")
        
        # 1. Test injury trend over time
        logger.info("Testing injury trend over time...")
        
        # First, let's verify our data
        logger.info("\nData verification:")
        logger.info(f"Unique seasons: {plays_data['season'].nunique()}")
        logger.info(f"Unique weeks: {plays_data['week'].nunique()}")
        logger.info(f"Total plays: {len(plays_data)}")
        logger.info(f"Total injuries: {len(injuries_data)}")
        
        # Calculate injury rate for trend plot
        time_data = plays_data[['season', 'week']].copy()
        time_data['game_id'] = time_data['season'].astype(int).astype(str) + '_' + time_data['week'].astype(int).astype(str)
        time_data['time_index'] = time_data['game_id'].str.replace('_', ' ')
        
        # Get total plays by time index
        time_stats = time_data.groupby('time_index').size().reset_index(name='total_plays')
        logger.info(f"\nTime stats shape: {time_stats.shape}")
        logger.info(f"Sample time stats:\n{time_stats.head()}")
        
        # Get injuries by time index
        time_injuries = injuries_data[['season', 'week']].copy()
        time_injuries['game_id'] = time_injuries['season'].astype(int).astype(str) + '_' + time_injuries['week'].astype(int).astype(str)
        time_injuries['time_index'] = time_injuries['game_id'].str.replace('_', ' ')
        time_injuries = time_injuries.groupby('time_index').size().reset_index(name='injury_count')
        logger.info(f"\nTime injuries shape: {time_injuries.shape}")
        logger.info(f"Sample time injuries:\n{time_injuries.head()}")
        
        # Merge and calculate rates
        time_data = pd.merge(time_stats, time_injuries, on='time_index', how='left')
        logger.info(f"\nMerged data shape: {time_data.shape}")
        logger.info(f"Sample merged data:\n{time_data.head()}")
        
        time_data['injury_count'] = time_data['injury_count'].fillna(0)
        time_data['injury_rate'] = (time_data['injury_count'] / time_data['total_plays']) * 1000
        
        # Sort by time index
        time_data = time_data.sort_values('time_index')
        
        # Print some statistics for verification
        logger.info("\nInjury rate statistics:")
        logger.info(f"Average injury rate: {time_data['injury_rate'].mean():.2f} injuries per 1000 plays")
        logger.info(f"Max injury rate: {time_data['injury_rate'].max():.2f} injuries per 1000 plays")
        logger.info(f"Min injury rate: {time_data['injury_rate'].min():.2f} injuries per 1000 plays")
        
        # Print full data for verification
        logger.info("\nFull time data:")
        logger.info(time_data)
        
        # Free memory
        del time_stats, time_injuries
        
        fig1 = plotter.plot_injury_trend_over_time(time_data)
        plt.show()
        input("Press Enter to continue...")
        
        # 2. Test injury by game situation
        logger.info("Testing injury by game situation...")
        
        # Calculate total injuries per game once for all analyses
        game_injuries = injuries_data.groupby(['season', 'week']).size().reset_index(name='injury_count')
        game_injuries['game_id'] = game_injuries['season'].astype(int).astype(str) + '_' + game_injuries['week'].astype(int).astype(str)
        
        # Add debug logging for game_id format
        logger.info("\nSample game_ids from plays data:")
        logger.info(plays_data['game_id'].head())
        logger.info("\nSample game_ids from injuries data:")
        logger.info(game_injuries['game_id'].head())
        
        # Update plays_data game_id to match
        plays_data['game_id'] = plays_data['season'].astype(int).astype(str) + '_' + plays_data['week'].astype(int).astype(str)
        
        # Quarter Analysis
        logger.info("Processing quarter analysis...")
        logger.info("Calculating plays by quarter...")
        quarter_plays = plays_data.groupby(['season', 'week', 'quarter']).size().reset_index(name='play_count')
        quarter_plays['game_id'] = quarter_plays['season'].astype(int).astype(str) + '_' + quarter_plays['week'].astype(int).astype(str)
        
        # Calculate distribution of plays by quarter within each game
        quarter_dist = quarter_plays.groupby(['game_id', 'quarter'])['play_count'].sum().reset_index()
        total_plays_per_game = quarter_dist.groupby('game_id')['play_count'].transform('sum')
        quarter_dist['play_fraction'] = quarter_dist['play_count'] / total_plays_per_game

        logger.info("Processing injuries by quarter...")
        # Add debug logging for merge
        logger.info("\nQuarter dist sample:")
        logger.info(quarter_dist.head())
        logger.info("\nGame injuries sample:")
        logger.info(game_injuries.head())
        
        # Merge play distribution with injuries to distribute injuries by quarter
        quarter_injuries = quarter_dist.merge(game_injuries[['game_id', 'injury_count']], on='game_id', how='left')
        logger.info("\nMerged quarter injuries sample:")
        logger.info(quarter_injuries.head())
        
        quarter_injuries['weighted_injuries'] = quarter_injuries['play_fraction'] * quarter_injuries['injury_count'].fillna(0)
        
        logger.info("Merging quarter data...")
        quarter_data = quarter_injuries.groupby('quarter').agg({
            'play_count': 'sum',
            'weighted_injuries': 'sum'
        }).reset_index()
        quarter_data['injury_rate'] = quarter_data['weighted_injuries'] / quarter_data['play_count'] * 1000
        
        # Down Analysis
        logger.info("Processing down analysis...")
        logger.info("Calculating plays by down...")
        down_plays = plays_data.groupby(['season', 'week', 'down']).size().reset_index(name='play_count')
        down_plays['game_id'] = down_plays['season'].astype(int).astype(str) + '_' + down_plays['week'].astype(int).astype(str)
        
        # Calculate distribution of plays by down within each game
        down_dist = down_plays.groupby(['game_id', 'down'])['play_count'].sum().reset_index()
        total_plays_per_game = down_dist.groupby('game_id')['play_count'].transform('sum')
        down_dist['play_fraction'] = down_dist['play_count'] / total_plays_per_game

        logger.info("Processing injuries by down...")
        # Merge play distribution with injuries to distribute injuries by down
        down_injuries = down_dist.merge(game_injuries[['game_id', 'injury_count']], on='game_id', how='left')
        down_injuries['weighted_injuries'] = down_injuries['play_fraction'] * down_injuries['injury_count'].fillna(0)
        
        logger.info("Merging down data...")
        down_data = down_injuries.groupby('down').agg({
            'play_count': 'sum',
            'weighted_injuries': 'sum'
        }).reset_index()
        down_data['injury_rate'] = down_data['weighted_injuries'] / down_data['play_count'] * 1000

        # Score differential analysis
        logger.info("Processing score differential analysis...")
        logger.info("Creating score differential bins...")
        plays_data['score_differential_bin'] = pd.cut(
            plays_data['score_differential'],
            bins=[-100, -21, -14, -7, 0, 7, 14, 21, 100],
            labels=['< -21', '-21 to -14', '-14 to -7', '-7 to 0', '0 to 7', '7 to 14', '14 to 21', '> 21']
        )
        
        # Calculate plays by score differential within each game
        logger.info("Calculating plays by score differential...")
        score_plays = plays_data.groupby(['season', 'week', 'score_differential_bin']).size().reset_index(name='play_count')
        score_plays['game_id'] = score_plays['season'].astype(int).astype(str) + '_' + score_plays['week'].astype(int).astype(str)
        
        # Calculate distribution of plays by score differential within each game
        score_dist = score_plays.groupby(['game_id', 'score_differential_bin'])['play_count'].sum().reset_index()
        total_plays_per_game = score_dist.groupby('game_id')['play_count'].transform('sum')
        score_dist['play_fraction'] = score_dist['play_count'] / total_plays_per_game

        logger.info("Processing injuries by score differential...")
        # Merge play distribution with injuries to distribute injuries by score differential
        score_injuries = score_dist.merge(game_injuries[['game_id', 'injury_count']], on='game_id', how='left')
        score_injuries['weighted_injuries'] = score_injuries['play_fraction'] * score_injuries['injury_count'].fillna(0)
        
        logger.info("Merging score differential data...")
        score_data = score_injuries.groupby('score_differential_bin').agg({
            'play_count': 'sum',
            'weighted_injuries': 'sum'
        }).reset_index()
        score_data['injury_rate'] = score_data['weighted_injuries'] / score_data['play_count'] * 1000
        
        # Time remaining analysis
        logger.info("Processing time remaining analysis...")
        logger.info("Creating time remaining bins...")
        plays_data['time_remaining_bin'] = pd.cut(
            plays_data['game_seconds_remaining'],
            bins=[0, 900, 1800, 2700, 3600],
            labels=['0-15 min', '15-30 min', '30-45 min', '45-60 min']
        )
        
        # Calculate plays by time remaining within each game
        logger.info("Calculating plays by time remaining...")
        time_plays = plays_data.groupby(['season', 'week', 'time_remaining_bin']).size().reset_index(name='play_count')
        time_plays['game_id'] = time_plays['season'].astype(int).astype(str) + '_' + time_plays['week'].astype(int).astype(str)
        
        # Calculate distribution of plays by time remaining within each game
        time_dist = time_plays.groupby(['game_id', 'time_remaining_bin'])['play_count'].sum().reset_index()
        total_plays_per_game = time_dist.groupby('game_id')['play_count'].transform('sum')
        time_dist['play_fraction'] = time_dist['play_count'] / total_plays_per_game

        logger.info("Processing injuries by time remaining...")
        # Merge play distribution with injuries to distribute injuries by time remaining
        time_injuries = time_dist.merge(game_injuries[['game_id', 'injury_count']], on='game_id', how='left')
        time_injuries['weighted_injuries'] = time_injuries['play_fraction'] * time_injuries['injury_count'].fillna(0)
        
        logger.info("Merging time remaining data...")
        time_data = time_injuries.groupby('time_remaining_bin').agg({
            'play_count': 'sum',
            'weighted_injuries': 'sum'
        }).reset_index()
        time_data['injury_rate'] = time_data['weighted_injuries'] / time_data['play_count'] * 1000
        
        # Add debug logging
        logger.info("\nGame injuries data:")
        logger.info(f"Total injuries: {game_injuries['injury_count'].sum()}")
        logger.info(f"Sample game injuries:\n{game_injuries.head()}")
        
        # Format data for visualization
        # Quarter data
        quarter_data['situation'] = 'quarter'
        quarter_data['value'] = quarter_data['quarter']
        
        # Down data
        down_data['situation'] = 'down'
        down_data['value'] = down_data['down']
        
        # Score data
        score_data['situation'] = 'score_differential'
        score_data['value'] = score_data['score_differential_bin']
        
        # Time data
        time_data['situation'] = 'time_remaining'
        time_data['value'] = time_data['time_remaining_bin']
        
        # Combine all data
        game_situation_data = pd.concat([
            quarter_data[['situation', 'value', 'play_count', 'weighted_injuries', 'injury_rate']],
            down_data[['situation', 'value', 'play_count', 'weighted_injuries', 'injury_rate']],
            score_data[['situation', 'value', 'play_count', 'weighted_injuries', 'injury_rate']],
            time_data[['situation', 'value', 'play_count', 'weighted_injuries', 'injury_rate']]
        ])
        
        # Log the formatted data
        logger.info("\nFormatted game situation data:")
        logger.info(game_situation_data)
        
        # Create visualization
        fig2 = plotter.plot_injury_by_game_situation(game_situation_data)
        plt.show()
        input("Press Enter to continue...")
        
        logger.info("Visualization testing complete!")
        
    except Exception as e:
        logger.error(f"Error during visualization testing: {str(e)}")
        raise

if __name__ == '__main__':
    main() 