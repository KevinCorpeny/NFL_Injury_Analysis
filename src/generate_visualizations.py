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
        
        # Create a unique identifier for each play
        plays_data['game_play_id'] = plays_data['game_id'] + '_' + plays_data['play_id'].astype(str)
        
        # Create injury flags for each play
        plays_data['injury_occurred'] = 0

        # Match injuries to specific plays using vectorized operations
        logger.info("🔍 Matching injuries to plays...")
        
        # Convert time values to be consistent
        plays_data['game_seconds_remaining'] = plays_data['quarter_seconds_remaining'] + (4 - plays_data['quarter']) * 900
        injuries_data['game_seconds_remaining'] = injuries_data['game_seconds_remaining'].astype(float)
        
        # Create a DataFrame with game-quarter combinations that have injuries
        injury_game_quarters = injuries_data[['season', 'week', 'quarter']].drop_duplicates()
        
        # Mark plays in quarters where injuries occurred
        plays_with_injuries = pd.merge(
            plays_data,
            injury_game_quarters,
            on=['season', 'week', 'quarter'],
            how='left',
            indicator=True
        )
        
        # Set injury_occurred to 1 for plays in quarters with injuries
        plays_data['injury_occurred'] = (plays_with_injuries['_merge'] == 'both').astype(int)
        
        total_matches = plays_data['injury_occurred'].sum()
        logger.info(f"✅ Total plays marked with injuries: {total_matches}")

        # Calculate injury rates by game situation
        logger.info("\n🔍 Analyzing injury rates by game situation...")
        
        # Filter for regular season weeks (1-18)
        plays_data = plays_data[plays_data['week'].between(1, 18)]
        
        # 1. Quarter analysis
        logger.info("📊 Calculating injury rates by quarter...")
        quarter_data = plays_data.groupby('quarter').agg({
            'play_id': 'count',
            'injury_occurred': 'sum'
        }).reset_index()
        quarter_data.columns = ['value', 'total_plays', 'injury_count']
        quarter_data['injury_rate'] = (quarter_data['injury_count'] / quarter_data['total_plays']) * 1000
        
        # 2. Down analysis
        logger.info("📊 Calculating injury rates by down...")
        down_data = plays_data[plays_data['down'].notna()].groupby('down').agg({
            'play_id': 'count',
            'injury_occurred': 'sum'
        }).reset_index()
        down_data.columns = ['value', 'total_plays', 'injury_count']
        down_data['injury_rate'] = (down_data['injury_count'] / down_data['total_plays']) * 1000
        
        # 3. Score differential analysis
        logger.info("📊 Calculating injury rates by score differential...")
        # Create score differential bins
        score_bins = [-50, -20, -10, -5, 0, 5, 10, 20, 50]
        bin_labels = [f"{score_bins[i]} to {score_bins[i+1]}" for i in range(len(score_bins)-1)]
        plays_data['score_bin'] = pd.cut(
            plays_data['score_differential'],
            bins=score_bins,
            labels=bin_labels,
            include_lowest=True
        )
        
        score_data = plays_data.groupby('score_bin').agg({
            'play_id': 'count',
            'injury_occurred': 'sum'
        }).reset_index()
        score_data.columns = ['value', 'total_plays', 'injury_count']
        score_data['injury_rate'] = (score_data['injury_count'] / score_data['total_plays']) * 1000
        
        # 4. Time remaining analysis
        logger.info("📊 Calculating injury rates by time remaining...")
        # Create time remaining bins (15-minute intervals)
        time_bins = [0, 900, 1800, 2700, 3600]
        time_labels = ['0-15', '15-30', '30-45', '45-60']
        plays_data['time_bin'] = pd.cut(
            plays_data['game_seconds_remaining'],
            bins=time_bins,
            labels=time_labels
        )
        
        time_data = plays_data.groupby('time_bin').agg({
            'play_id': 'count',
            'injury_occurred': 'sum'
        }).reset_index()
        time_data.columns = ['value', 'total_plays', 'injury_count']
        time_data['injury_rate'] = (time_data['injury_count'] / time_data['total_plays']) * 1000
        
        # Log statistics
        logger.info("\n📊 Game Situation Statistics")
        
        # Format and log statistics for each category
        format_statistics(quarter_data.rename(columns={'value': 'quarter'}), "Quarter Statistics")
        format_statistics(down_data.rename(columns={'value': 'down'}), "Down Statistics")
        format_statistics(score_data.rename(columns={'value': 'score_bin'}), "Score Differential Statistics")
        format_statistics(time_data.rename(columns={'value': 'time_bin'}), "Time Remaining Statistics")
        
        # Calculate injury rate for trend plot
        logger.info("\n🎨 Preparing data for visualization...")
        
        # First, get all valid game dates from plays data
        plays_by_date = plays_data.groupby(['season', 'week']).agg({
            'game_id': 'nunique',  # Count unique games
            'play_id': 'count'     # Count total plays
        }).reset_index()
        plays_by_date.columns = ['season', 'week', 'num_games', 'total_plays']
        
        # Get injuries by date
        injuries_by_date = injuries_data.groupby(['season', 'week']).size().reset_index(name='injury_count')
        
        # Merge plays and injuries data
        trend_data = plays_by_date.merge(
            injuries_by_date,
            on=['season', 'week'],
            how='left'
        )
        
        # Fill missing injury counts with 0
        trend_data['injury_count'] = trend_data['injury_count'].fillna(0)
        
        # Create time index for display
        trend_data['time_index'] = trend_data['season'].astype(str) + ' ' + trend_data['week'].astype(str).str.zfill(2)
        
        # Calculate injury rate per 1000 plays
        trend_data['injury_rate'] = (trend_data['injury_count'] / trend_data['total_plays']) * 1000
        
        # Sort by season and week
        trend_data = trend_data.sort_values(['season', 'week'])
        
        # Log some statistics for verification
        logger.info("\nTime series statistics:")
        logger.info(f"Total weeks: {len(trend_data)}")
        logger.info(f"Total plays: {trend_data['total_plays'].sum():,}")
        logger.info(f"Total injuries: {trend_data['injury_count'].sum():,}")
        logger.info(f"Average games per week: {trend_data['num_games'].mean():.1f}")
        logger.info(f"Average plays per week: {trend_data['total_plays'].mean():.1f}")
        logger.info(f"Average injuries per week: {trend_data['injury_count'].mean():.1f}")
        
        # Create plot data for game situation plots
        # Create separate dataframes for each situation type
        quarter_plot_data = quarter_data.copy()
        quarter_plot_data = quarter_plot_data.rename(columns={'value': 'quarter'})
        
        down_plot_data = down_data.copy()
        down_plot_data = down_plot_data.rename(columns={'value': 'down'})
        
        score_plot_data = score_data.copy()
        score_plot_data = score_plot_data.rename(columns={'value': 'score_bin'})
        
        time_plot_data = time_data.copy()
        time_plot_data = time_plot_data.rename(columns={'value': 'time_bin'})
        
        # Generate visualizations
        logger.info("\n🖼️ Generating visualizations...")
        
        # Plot injury trends over time
        logger.info("  📈 Creating injury trend plot...")
        fig_trend = plotter.plot_injury_trend_over_time(trend_data)
        plotter.save_plot('injury_trends.png')
        
        # Plot injury rates by game situation
        logger.info("  📊 Creating game situation plots...")
        fig_situation = plotter.plot_injury_by_game_situation(
            quarter_plot_data,
            down_plot_data,
            score_plot_data,
            time_plot_data
        )
        plotter.save_plot('injury_by_game_situation.png')
        
        logger.info("\n✨ Visualization generation complete!")
        logger.info(f"📁 Plots saved to: {output_dir}")
        
    except Exception as e:
        logger.error(f"❌ Error generating visualizations: {str(e)}")
        raise

if __name__ == "__main__":
    main() 