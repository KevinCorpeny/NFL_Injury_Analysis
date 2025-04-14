from pathlib import Path
import pandas as pd
import numpy as np
from src.visualization.plotter import NFLPlotter
from src.utils.logging import logger, setup_logging
import matplotlib.pyplot as plt
import seaborn as sns

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
                logger.info(f"{col:>20}: {row[col]:<30}")
            elif col in ['score_differential', 'game_seconds_remaining']:
                logger.info(f"{col:>20}: {row[col]:<15}")
            elif col in ['quarter', 'down']:
                logger.info(f"{col:>20}: {row[col]:<15}")
            else:
                logger.info(f"{col:>20}: {row[col]:<15}")
        logger.info("-"*80)

def main():
    """Generate visualizations for NFL injury analysis."""
    try:
        # Set up logging
        setup_logging()
        
        # Create output directory
        output_dir = Path("reports/figures")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize plotter
        plotter = NFLPlotter(output_dir)
        
        # Load processed data
        logger.info("📊 Loading processed NFL data...")
        plays_data = pd.read_parquet("data/processed/processed_plays.parquet")
        injuries_data = pd.read_parquet("data/processed/processed_injuries.parquet")
        
        logger.info(f"✅ Successfully loaded {len(plays_data):,} plays and {len(injuries_data):,} injuries")
        
        # Calculate injury rates by game situation
        logger.info("\n🔍 Analyzing injury rates by game situation...")
        
        # Create a unique identifier for each game
        plays_data['game_id'] = plays_data['season'].astype(str) + '_' + plays_data['week'].astype(str)
        injuries_data['game_id'] = injuries_data['season'].astype(str) + '_' + injuries_data['week'].astype(str)
        
        # 1. Quarter analysis
        logger.info("📊 Calculating injury rates by quarter...")
        
        # Calculate total plays and injuries per week/season
        plays_by_week = plays_data.groupby(['season', 'week']).size().reset_index(name='total_plays_week')
        injuries_by_week = injuries_data.groupby(['season', 'week']).size().reset_index(name='total_injuries_week')
        
        # Calculate plays by quarter for each week/season
        quarter_by_week = plays_data.groupby(['season', 'week', 'quarter']).size().reset_index(name='plays_in_quarter')
        
        # Merge with total plays and injuries
        quarter_merged = quarter_by_week.merge(plays_by_week, on=['season', 'week'])
        quarter_merged = quarter_merged.merge(injuries_by_week, on=['season', 'week'])
        
        # Calculate proportion of plays in each quarter and distribute injuries
        quarter_merged['proportion'] = quarter_merged['plays_in_quarter'] / quarter_merged['total_plays_week']
        quarter_merged['injuries'] = quarter_merged['proportion'] * quarter_merged['total_injuries_week']
        
        # Aggregate by quarter
        quarter_data = quarter_merged.groupby('quarter').agg({
            'plays_in_quarter': 'sum',
            'injuries': 'sum'
        }).reset_index()
        
        quarter_data.columns = ['quarter', 'total_plays', 'injury_count']
        quarter_data['injury_rate'] = (quarter_data['injury_count'] / quarter_data['total_plays']) * 1000
        
        # 2. Down analysis
        logger.info("📊 Calculating injury rates by down...")
        
        # Calculate plays by down for each week/season
        down_by_week = plays_data.groupby(['season', 'week', 'down']).size().reset_index(name='plays_in_down')
        
        # Merge with total plays and injuries
        down_merged = down_by_week.merge(plays_by_week, on=['season', 'week'])
        down_merged = down_merged.merge(injuries_by_week, on=['season', 'week'])
        
        # Calculate proportion of plays in each down and distribute injuries
        down_merged['proportion'] = down_merged['plays_in_down'] / down_merged['total_plays_week']
        down_merged['injuries'] = down_merged['proportion'] * down_merged['total_injuries_week']
        
        # Aggregate by down
        down_data = down_merged.groupby('down').agg({
            'plays_in_down': 'sum',
            'injuries': 'sum'
        }).reset_index()
        
        down_data.columns = ['down', 'total_plays', 'injury_count']
        down_data['injury_rate'] = (down_data['injury_count'] / down_data['total_plays']) * 1000
        
        # 3. Score differential analysis
        logger.info("📊 Calculating injury rates by score differential...")
        
        # Create score differential bins
        score_bins = [-50, -20, -10, -5, 0, 5, 10, 20, 50]
        plays_data['score_differential_bin'] = pd.cut(
            plays_data['score_differential'],
            bins=score_bins,
            labels=[f"{score_bins[i]} to {score_bins[i+1]}" for i in range(len(score_bins)-1)],
            include_lowest=True
        )
        
        # Calculate plays by score differential for each week/season
        score_by_week = plays_data.groupby(['season', 'week', 'score_differential_bin']).size().reset_index(name='plays_in_score')
        
        # Merge with total plays and injuries
        score_merged = score_by_week.merge(plays_by_week, on=['season', 'week'])
        score_merged = score_merged.merge(injuries_by_week, on=['season', 'week'])
        
        # Calculate proportion of plays in each score differential and distribute injuries
        score_merged['proportion'] = score_merged['plays_in_score'] / score_merged['total_plays_week']
        score_merged['injuries'] = score_merged['proportion'] * score_merged['total_injuries_week']
        
        # Aggregate by score differential
        score_data = score_merged.groupby('score_differential_bin').agg({
            'plays_in_score': 'sum',
            'injuries': 'sum'
        }).reset_index()
        
        score_data.columns = ['score_differential_bin', 'total_plays', 'injury_count']
        score_data['injury_rate'] = (score_data['injury_count'] / score_data['total_plays']) * 1000
        
        # 4. Time remaining analysis
        logger.info("📊 Calculating injury rates by time remaining...")
        
        # Create time remaining bins
        time_bins = [0, 900, 1800, 2700, 3600]  # 15-minute intervals
        plays_data['time_remaining_bin'] = pd.cut(
            plays_data['game_seconds_remaining'],
            bins=time_bins,
            labels=['0-15', '15-30', '30-45', '45-60']
        )
        
        # Calculate plays by time remaining for each week/season
        time_by_week = plays_data.groupby(['season', 'week', 'time_remaining_bin']).size().reset_index(name='plays_in_time')
        
        # Merge with total plays and injuries
        time_merged = time_by_week.merge(plays_by_week, on=['season', 'week'])
        time_merged = time_merged.merge(injuries_by_week, on=['season', 'week'])
        
        # Calculate proportion of plays in each time bin and distribute injuries
        time_merged['proportion'] = time_merged['plays_in_time'] / time_merged['total_plays_week']
        time_merged['injuries'] = time_merged['proportion'] * time_merged['total_injuries_week']
        
        # Aggregate by time remaining
        time_data = time_merged.groupby('time_remaining_bin').agg({
            'plays_in_time': 'sum',
            'injuries': 'sum'
        }).reset_index()
        
        time_data.columns = ['time_remaining_bin', 'total_plays', 'injury_count']
        time_data['injury_rate'] = (time_data['injury_count'] / time_data['total_plays']) * 1000
        
        # Log statistics in a readable format
        logger.info("\n📊 Game Situation Statistics")
        
        # Format and log quarter statistics
        logger.info("\nQuarter Statistics:")
        format_statistics(quarter_data, "Quarter Statistics")
        
        # Format and log down statistics
        logger.info("\nDown Statistics:")
        format_statistics(down_data, "Down Statistics")
        
        # Format and log score differential statistics
        logger.info("\nScore Differential Statistics:")
        format_statistics(score_data, "Score Differential Statistics")
        
        # Format and log time remaining statistics
        logger.info("\nTime Remaining Statistics:")
        format_statistics(time_data, "Time Remaining Statistics")
        
        # Create final dataset for plotting
        logger.info("\n🎨 Preparing data for visualization...")
        
        # Calculate injury rate for trend plot
        trend_data = plays_data.groupby(['season', 'week']).size().reset_index(name='total_plays')
        
        # Create time index that can be properly converted to numeric values
        trend_data['time_index'] = trend_data['season'].astype(str) + ' ' + trend_data['week'].astype(str).str.zfill(2)
        
        # Get injuries by time index
        trend_injuries = injuries_data.groupby(['season', 'week']).size().reset_index(name='injury_count')
        
        # Merge and calculate rates
        trend_data = pd.merge(
            trend_data,
            trend_injuries[['season', 'week', 'injury_count']],
            on=['season', 'week'],
            how='left'
        )
        trend_data['injury_count'] = trend_data['injury_count'].fillna(0)
        trend_data['injury_rate'] = (trend_data['injury_count'] / trend_data['total_plays']) * 1000
        
        # Sort by time index
        trend_data = trend_data.sort_values(['season', 'week'])
        
        # Free memory
        del trend_injuries
        
        # Create plot data for game situation plots
        # Create separate dataframes for each situation type
        quarter_plot_data = quarter_data.copy()
        quarter_plot_data['situation'] = 'quarter'
        quarter_plot_data['value'] = quarter_plot_data['quarter']
        quarter_plot_data['injury_rate'] = quarter_plot_data['injury_rate']
        quarter_plot_data['play_count'] = quarter_plot_data['total_plays']
        
        down_plot_data = down_data.copy()
        down_plot_data['situation'] = 'down'
        down_plot_data['value'] = down_plot_data['down']
        down_plot_data['injury_rate'] = down_plot_data['injury_rate']
        down_plot_data['play_count'] = down_plot_data['total_plays']
        
        score_plot_data = score_data.copy()
        score_plot_data['situation'] = 'score_differential'
        score_plot_data['value'] = score_plot_data['score_differential_bin']
        score_plot_data['injury_rate'] = score_plot_data['injury_rate']
        score_plot_data['play_count'] = score_plot_data['total_plays']
        
        time_plot_data = time_data.copy()
        time_plot_data['situation'] = 'time_remaining'
        time_plot_data['value'] = time_plot_data['time_remaining_bin']
        time_plot_data['injury_rate'] = time_plot_data['injury_rate']
        time_plot_data['play_count'] = time_plot_data['total_plays']
        
        # Combine all situation data
        plot_data = pd.concat([
            quarter_plot_data[['situation', 'value', 'injury_rate', 'play_count']],
            down_plot_data[['situation', 'value', 'injury_rate', 'play_count']],
            score_plot_data[['situation', 'value', 'injury_rate', 'play_count']],
            time_plot_data[['situation', 'value', 'injury_rate', 'play_count']]
        ], ignore_index=True)
        
        # Generate visualizations
        logger.info("\n🖼️ Generating visualizations...")
        
        # Plot injury trends over time
        logger.info("  📈 Creating injury trend plot...")
        fig_trend = plotter.plot_injury_trend_over_time(trend_data)
        plotter.save_plot('injury_trends.png')
        
        # Plot injury rates by game situation
        logger.info("  📊 Creating game situation plots...")
        fig_situation = plotter.plot_injury_by_game_situation(plot_data)
        plotter.save_plot('injury_by_situation.png')
        
        logger.info("\n✨ Visualization generation complete!")
        logger.info(f"📁 Plots saved to: {output_dir}")
        
    except Exception as e:
        logger.error(f"❌ Error generating visualizations: {str(e)}")
        raise

if __name__ == "__main__":
    main() 