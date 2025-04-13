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
        plays_data['game_id'] = plays_data['season'].astype(str) + '_' + plays_data['week'].astype(str)
        injuries_data['game_id'] = injuries_data['season'].astype(str) + '_' + injuries_data['week'].astype(str)
        
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
        
        # Process game situations one at a time to save memory
        game_situation_data = pd.DataFrame()
        
        # 1. Quarter analysis
        logger.info("Processing quarter data...")
        
        # Get total plays by quarter
        quarter_stats = plays_data.groupby('quarter').size().reset_index(name='total_plays')
        
        # Get all quarters that occur in each game
        game_quarters = plays_data.groupby(['season', 'week'])['quarter'].unique().reset_index()
        # Explode the quarters array to get one row per quarter per game
        game_quarters = game_quarters.explode('quarter')
        
        # Now merge with injuries to distribute injuries across quarters
        injuries_by_game = injuries_data.groupby(['season', 'week']).size().reset_index(name='injury_count')
        injuries_with_quarter = injuries_by_game.merge(game_quarters, on=['season', 'week'], how='inner')
        # Distribute injuries evenly across quarters in each game
        injuries_with_quarter['injury_count'] = injuries_with_quarter['injury_count'] / injuries_with_quarter.groupby(['season', 'week'])['quarter'].transform('count')
        
        # Sum up injuries by quarter
        quarter_injuries = injuries_with_quarter.groupby('quarter')['injury_count'].sum().reset_index()
        
        # Merge and calculate rates
        quarter_data = pd.merge(quarter_stats, quarter_injuries, on='quarter', how='left')
        quarter_data['injury_count'] = quarter_data['injury_count'].fillna(0)
        quarter_data['injury_rate'] = (quarter_data['injury_count'] / quarter_data['total_plays']) * 1000
        
        # Store in game situation data
        game_situation_data['quarter'] = quarter_data['quarter'].astype(int)
        game_situation_data['injury_rate_quarter'] = quarter_data['injury_rate']
        
        # Print quarter statistics for verification
        logger.info("\nQuarter statistics:")
        format_statistics(quarter_data, "Quarter Statistics")
        
        # Free memory
        del quarter_stats, quarter_injuries, injuries_with_quarter, game_quarters, injuries_by_game
        
        # 2. Down analysis
        logger.info("Processing down data...")
        
        # Get total plays by down
        down_stats = plays_data.groupby('down').size().reset_index(name='total_plays')
        
        # Get all downs that occur in each game
        game_downs = plays_data.groupby(['season', 'week'])['down'].unique().reset_index()
        # Explode the downs array to get one row per down per game
        game_downs = game_downs.explode('down')
        
        # Now merge with injuries to distribute injuries across downs
        injuries_by_game = injuries_data.groupby(['season', 'week']).size().reset_index(name='injury_count')
        injuries_with_down = injuries_by_game.merge(game_downs, on=['season', 'week'], how='inner')
        # Distribute injuries evenly across downs in each game
        injuries_with_down['injury_count'] = injuries_with_down['injury_count'] / injuries_with_down.groupby(['season', 'week'])['down'].transform('count')
        
        # Sum up injuries by down
        down_injuries = injuries_with_down.groupby('down')['injury_count'].sum().reset_index()
        
        # Merge and calculate rates
        down_data = pd.merge(down_stats, down_injuries, on='down', how='left')
        down_data['injury_count'] = down_data['injury_count'].fillna(0)
        down_data['injury_rate'] = (down_data['injury_count'] / down_data['total_plays']) * 1000
        
        # Store in game situation data
        game_situation_data['down'] = down_data['down'].astype(int)
        game_situation_data['injury_rate_down'] = down_data['injury_rate']
        
        # Print down statistics for verification
        logger.info("\nDown statistics:")
        format_statistics(down_data, "Down Statistics")
        
        # Free memory
        del down_stats, down_injuries, injuries_with_down, game_downs, injuries_by_game
        
        # 3. Score differential analysis
        logger.info("Processing score differential data...")
        
        # Create score differential bins using pd.cut with custom bins
        score_bins = [-50, -20, -10, -5, 0, 5, 10, 20, 50]
        plays_data['score_differential_bin'] = pd.cut(
            plays_data['score_differential'],
            bins=score_bins,
            labels=[f"{score_bins[i]} to {score_bins[i+1]}" for i in range(len(score_bins)-1)],
            include_lowest=True
        )
        
        # Get total plays by score differential bin
        score_stats = plays_data.groupby('score_differential_bin').size().reset_index(name='total_plays')
        
        # Get all score bins that occur in each game
        game_scores = plays_data.groupby(['season', 'week'])['score_differential_bin'].unique().reset_index()
        # Explode the score bins array to get one row per bin per game
        game_scores = game_scores.explode('score_differential_bin')
        
        # Now merge with injuries to distribute injuries across score bins
        injuries_by_game = injuries_data.groupby(['season', 'week']).size().reset_index(name='injury_count')
        injuries_with_score = injuries_by_game.merge(game_scores, on=['season', 'week'], how='inner')
        # Distribute injuries evenly across score bins in each game
        injuries_with_score['injury_count'] = injuries_with_score['injury_count'] / injuries_with_score.groupby(['season', 'week'])['score_differential_bin'].transform('count')
        
        # Sum up injuries by score differential bin
        score_injuries = injuries_with_score.groupby('score_differential_bin')['injury_count'].sum().reset_index()
        
        # Merge and calculate rates
        score_data = pd.merge(score_stats, score_injuries, on='score_differential_bin', how='left')
        score_data['injury_count'] = score_data['injury_count'].fillna(0)
        score_data['injury_rate'] = (score_data['injury_count'] / score_data['total_plays']) * 1000
        
        # Store in game situation data
        game_situation_data['score_differential'] = score_data['score_differential_bin']
        game_situation_data['score_differential_numeric'] = score_data['score_differential_bin'].apply(
            lambda x: float(x.split(' to ')[0]) + (float(x.split(' to ')[1]) - float(x.split(' to ')[0])) / 2
        )
        game_situation_data['injury_rate_score'] = score_data['injury_rate']
        
        # Print score differential statistics for verification
        logger.info("\nScore differential statistics:")
        format_statistics(score_data, "Score Differential Statistics")
        
        # Free memory
        del score_stats, score_injuries, injuries_with_score, game_scores, injuries_by_game
        
        # 4. Time remaining analysis
        logger.info("Processing time remaining data...")
        
        # Create time remaining bins using pd.cut with custom bins
        time_bins = [0, 900, 1800, 2700, 3600]  # 15-minute intervals
        plays_data['time_remaining_bin'] = pd.cut(
            plays_data['game_seconds_remaining'],
            bins=time_bins,
            labels=['0-15', '15-30', '30-45', '45-60']
        )
        
        # Get total plays by time remaining bin
        time_stats = plays_data.groupby('time_remaining_bin').size().reset_index(name='total_plays')
        
        # Get injuries by time remaining bin
        # First get all time remaining bins that occur in each game
        game_times = plays_data.groupby(['season', 'week'])['time_remaining_bin'].unique().reset_index()
        # Explode the time bins array to get one row per bin per game
        game_times = game_times.explode('time_remaining_bin')
        
        # Now merge with injuries to distribute injuries across time bins
        injuries_by_game = injuries_data.groupby(['season', 'week']).size().reset_index(name='injury_count')
        injuries_with_time = injuries_by_game.merge(game_times, on=['season', 'week'], how='inner')
        # Distribute injuries evenly across time bins in each game
        injuries_with_time['injury_count'] = injuries_with_time['injury_count'] / injuries_with_time.groupby(['season', 'week'])['time_remaining_bin'].transform('count')
        
        # Sum up injuries by time remaining bin
        time_injuries = injuries_with_time.groupby('time_remaining_bin')['injury_count'].sum().reset_index()
        
        # Merge and calculate rates
        time_data = pd.merge(time_stats, time_injuries, on='time_remaining_bin', how='left')
        time_data['injury_count'] = time_data['injury_count'].fillna(0)
        time_data['injury_rate'] = (time_data['injury_count'] / time_data['total_plays']) * 1000
        
        # Store in game situation data
        game_situation_data['time_remaining'] = time_data['time_remaining_bin'].astype(str)
        game_situation_data['time_remaining_numeric'] = time_data['time_remaining_bin'].apply(
            lambda x: float(x.split('-')[0]) + (float(x.split('-')[1]) - float(x.split('-')[0])) / 2 if isinstance(x, str) else np.nan
        )
        game_situation_data['injury_rate_time'] = time_data['injury_rate']
        
        # Print time remaining statistics for verification
        logger.info("\nTime remaining statistics:")
        format_statistics(time_data, "Time Remaining Statistics")
        
        # Free memory
        del time_stats, time_injuries, injuries_with_time, game_times, injuries_by_game
        
        # Print final game situation data for verification
        logger.info("\nFinal game situation data:")
        logger.info(game_situation_data)
        
        fig2 = plotter.plot_injury_by_game_situation(game_situation_data)
        plt.show()
        input("Press Enter to continue...")
        
        logger.info("Visualization testing complete!")
        
    except Exception as e:
        logger.error(f"Error during visualization testing: {str(e)}")
        raise

if __name__ == '__main__':
    main() 