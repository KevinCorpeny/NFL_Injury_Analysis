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
        
        # Calculate injury rates by quarter
        logger.info("  📊 Calculating injury rates by quarter...")
        quarter_data = plays_data.groupby('quarter').agg({
            'play_count': 'sum'
        }).reset_index()
        quarter_injuries = injuries_data.groupby('quarter').size().reset_index(name='injury_count')
        quarter_data = pd.merge(quarter_data, quarter_injuries, on='quarter', how='left')
        quarter_data['injury_count'] = quarter_data['injury_count'].fillna(0)
        quarter_data['injury_rate_quarter'] = (quarter_data['injury_count'] / quarter_data['play_count']) * 1000
        
        # Calculate injury rates by down
        logger.info("  📊 Calculating injury rates by down...")
        down_data = plays_data.groupby('down').agg({
            'play_count': 'sum'
        }).reset_index()
        down_injuries = injuries_data.groupby('down').size().reset_index(name='injury_count')
        down_data = pd.merge(down_data, down_injuries, on='down', how='left')
        down_data['injury_count'] = down_data['injury_count'].fillna(0)
        down_data['injury_rate_down'] = (down_data['injury_count'] / down_data['play_count']) * 1000
        
        # Calculate injury rates by score differential
        logger.info("  📊 Calculating injury rates by score differential...")
        # Create bins for score differential
        score_bins = pd.cut(plays_data['score_differential'], bins=10)
        score_data = plays_data.groupby(score_bins).agg({
            'play_count': 'sum'
        }).reset_index()
        score_data.rename(columns={'score_differential': 'score_bin'}, inplace=True)
        
        score_injuries = injuries_data.merge(
            plays_data[['season', 'week', 'score_differential']],
            on=['season', 'week'],
            how='left'
        )
        score_injuries = score_injuries.groupby(pd.cut(score_injuries['score_differential'], bins=10)).size().reset_index(name='injury_count')
        score_injuries.rename(columns={'score_differential': 'score_bin'}, inplace=True)
        
        score_data = pd.merge(score_data, score_injuries, on='score_bin', how='left')
        score_data['injury_count'] = score_data['injury_count'].fillna(0)
        score_data['injury_rate_score'] = (score_data['injury_count'] / score_data['play_count']) * 1000
        
        # Calculate injury rates by time remaining
        logger.info("  📊 Calculating injury rates by time remaining...")
        # Create bins for time remaining
        time_bins = pd.cut(plays_data['game_seconds_remaining'], bins=10)
        time_data = plays_data.groupby(time_bins).agg({
            'play_count': 'sum'
        }).reset_index()
        time_data.rename(columns={'game_seconds_remaining': 'time_bin'}, inplace=True)
        
        time_injuries = injuries_data.merge(
            plays_data[['season', 'week', 'game_seconds_remaining']],
            on=['season', 'week'],
            how='left'
        )
        time_injuries = time_injuries.groupby(pd.cut(time_injuries['game_seconds_remaining'], bins=10)).size().reset_index(name='injury_count')
        time_injuries.rename(columns={'game_seconds_remaining': 'time_bin'}, inplace=True)
        
        time_data = pd.merge(time_data, time_injuries, on='time_bin', how='left')
        time_data['injury_count'] = time_data['injury_count'].fillna(0)
        time_data['injury_rate_time'] = (time_data['injury_count'] / time_data['play_count']) * 1000
        
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
        plot_data = pd.merge(
            plays_data,
            quarter_data[['quarter', 'injury_rate_quarter']],
            on='quarter',
            how='left'
        )
        plot_data = pd.merge(
            plot_data,
            down_data[['down', 'injury_rate_down']],
            on='down',
            how='left'
        )
        plot_data = pd.merge(
            plot_data,
            score_data[['score_bin', 'injury_rate_score']],
            left_on=pd.cut(plot_data['score_differential'], bins=10),
            right_on='score_bin',
            how='left'
        )
        plot_data = pd.merge(
            plot_data,
            time_data[['time_bin', 'injury_rate_time']],
            left_on=pd.cut(plot_data['game_seconds_remaining'], bins=10),
            right_on='time_bin',
            how='left'
        )
        
        # Generate visualizations
        logger.info("\n🖼️ Generating visualizations...")
        
        # Plot injury trends over time
        logger.info("  📈 Creating injury trend plot...")
        fig_trend = plotter.plot_injury_trend_over_time(plot_data)
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