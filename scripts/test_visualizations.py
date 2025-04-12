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

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.visualization.plotter import NFLPlotter
from src.utils.logging import setup_logging, logger

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
        plays_cols = ['season', 'week', 'quarter', 'down', 'score_differential', 'game_seconds_remaining']
        injuries_cols = ['season', 'week']
        
        logger.info(f"Loading data from {plays_path} and {injuries_path}")
        plays_data = pd.read_parquet(plays_path, columns=plays_cols)
        injuries_data = pd.read_parquet(injuries_path, columns=injuries_cols)
        
        logger.info(f"Loaded {len(plays_data)} plays and {len(injuries_data)} injuries")
        
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
        
        # Aggregate plays and injuries by season and week
        play_counts = plays_data.groupby(['season', 'week']).size().reset_index(name='play_count')
        injury_counts = injuries_data.groupby(['season', 'week']).size().reset_index(name='injury_count')
        
        # Merge and calculate injury rate
        time_data = pd.merge(play_counts, injury_counts, on=['season', 'week'], how='left')
        time_data['injury_count'] = time_data['injury_count'].fillna(0)
        time_data['injury_rate'] = (time_data['injury_count'] / time_data['play_count']) * 1000
        
        # Sort by season and week for proper time series
        time_data = time_data.sort_values(['season', 'week'])
        
        # Print some statistics for verification
        logger.info("\nInjury rate statistics:")
        logger.info(f"Average injury rate: {time_data['injury_rate'].mean():.2f} injuries per 1000 plays")
        logger.info(f"Max injury rate: {time_data['injury_rate'].max():.2f} injuries per 1000 plays")
        logger.info(f"Min injury rate: {time_data['injury_rate'].min():.2f} injuries per 1000 plays")
        
        # Free up memory
        del play_counts, injury_counts
        
        fig1 = plotter.plot_injury_trend_over_time(time_data)
        plt.show()
        input("Press Enter to continue...")
        
        # 2. Test injury by game situation
        logger.info("Testing injury by game situation...")
        
        # Process game situations one at a time to save memory
        game_situation_data = pd.DataFrame()
        
        # 1. Quarter analysis
        logger.info("Processing quarter data...")
        # First, get total plays by quarter
        quarter_stats = plays_data.groupby('quarter').size().reset_index(name='total_plays')
        
        # Then, get injuries by quarter by merging with plays
        quarter_injuries = injuries_data.merge(
            plays_data[['season', 'week', 'quarter']],
            on=['season', 'week'],
            how='left'
        )
        quarter_injuries = quarter_injuries.groupby('quarter').size().reset_index(name='injury_count')
        
        # Merge and calculate rates
        quarter_data = pd.merge(quarter_stats, quarter_injuries, on='quarter', how='left')
        quarter_data['injury_count'] = quarter_data['injury_count'].fillna(0)
        quarter_data['injury_rate'] = (quarter_data['injury_count'] / quarter_data['total_plays']) * 1000
        
        # Store in game situation data
        game_situation_data['quarter'] = quarter_data['quarter']
        game_situation_data['injury_rate_quarter'] = quarter_data['injury_rate']
        
        # Print quarter statistics for verification
        logger.info("\nQuarter statistics:")
        logger.info(quarter_data)
        
        # Free memory
        del quarter_stats, quarter_injuries, quarter_data
        
        # 2. Down analysis
        logger.info("Processing down data...")
        # Get total plays by down
        down_stats = plays_data.groupby('down').size().reset_index(name='total_plays')
        
        # Get injuries by down
        down_injuries = injuries_data.merge(
            plays_data[['season', 'week', 'down']],
            on=['season', 'week'],
            how='left'
        )
        down_injuries = down_injuries.groupby('down').size().reset_index(name='injury_count')
        
        # Merge and calculate rates
        down_data = pd.merge(down_stats, down_injuries, on='down', how='left')
        down_data['injury_count'] = down_data['injury_count'].fillna(0)
        down_data['injury_rate'] = (down_data['injury_count'] / down_data['total_plays']) * 1000
        
        # Store in game situation data
        game_situation_data['down'] = down_data['down']
        game_situation_data['injury_rate_down'] = down_data['injury_rate']
        
        # Print down statistics for verification
        logger.info("\nDown statistics:")
        logger.info(down_data)
        
        # Free memory
        del down_stats, down_injuries, down_data
        
        # 3. Score differential analysis
        logger.info("Processing score differential data...")
        # Create bins for score differential
        score_bins = pd.cut(plays_data['score_differential'], bins=10)
        score_stats = plays_data.groupby(score_bins).size().reset_index(name='total_plays')
        
        # Get injuries by score differential
        score_injuries = injuries_data.merge(
            plays_data[['season', 'week', 'score_differential']],
            on=['season', 'week'],
            how='left'
        )
        score_injuries['score_bin'] = pd.cut(score_injuries['score_differential'], bins=10)
        score_injuries = score_injuries.groupby('score_bin').size().reset_index(name='injury_count')
        
        # Merge and calculate rates
        score_data = pd.merge(score_stats, score_injuries, left_on='score_differential', right_on='score_bin', how='left')
        score_data['injury_count'] = score_data['injury_count'].fillna(0)
        score_data['injury_rate'] = (score_data['injury_count'] / score_data['total_plays']) * 1000
        
        # Store in game situation data
        game_situation_data['score_differential'] = score_data['score_differential'].apply(lambda x: x.mid)
        game_situation_data['injury_rate_score'] = score_data['injury_rate']
        
        # Print score differential statistics for verification
        logger.info("\nScore differential statistics:")
        logger.info(score_data)
        
        # Free memory
        del score_stats, score_injuries, score_data
        
        # 4. Time remaining analysis
        logger.info("Processing time remaining data...")
        # Create bins for time remaining
        time_bins = pd.cut(plays_data['game_seconds_remaining'], bins=10)
        time_stats = plays_data.groupby(time_bins).size().reset_index(name='total_plays')
        
        # Get injuries by time remaining
        time_injuries = injuries_data.merge(
            plays_data[['season', 'week', 'game_seconds_remaining']],
            on=['season', 'week'],
            how='left'
        )
        time_injuries['time_bin'] = pd.cut(time_injuries['game_seconds_remaining'], bins=10)
        time_injuries = time_injuries.groupby('time_bin').size().reset_index(name='injury_count')
        
        # Merge and calculate rates
        time_data = pd.merge(time_stats, time_injuries, left_on='game_seconds_remaining', right_on='time_bin', how='left')
        time_data['injury_count'] = time_data['injury_count'].fillna(0)
        time_data['injury_rate'] = (time_data['injury_count'] / time_data['total_plays']) * 1000
        
        # Store in game situation data
        game_situation_data['game_seconds_remaining'] = time_data['game_seconds_remaining'].apply(lambda x: x.mid)
        game_situation_data['injury_rate_time'] = time_data['injury_rate']
        
        # Print time remaining statistics for verification
        logger.info("\nTime remaining statistics:")
        logger.info(time_data)
        
        # Free memory
        del time_stats, time_injuries, time_data
        
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