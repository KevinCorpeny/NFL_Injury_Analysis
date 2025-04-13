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
        # Get total plays by quarter
        quarter_stats = plays_data.groupby('quarter').size().reset_index(name='total_plays')
        
        # Get injuries by game and quarter
        injuries_with_quarter = injuries_data[['game_id']].merge(
            plays_data[['game_id', 'quarter']],
            on='game_id',
            how='left'
        )
        
        # Count injuries by quarter
        quarter_injuries = injuries_with_quarter.groupby('quarter').size().reset_index(name='injury_count')
        
        # Merge and calculate rates
        quarter_data = pd.merge(quarter_stats, quarter_injuries, on='quarter', how='left')
        quarter_data['injury_count'] = quarter_data['injury_count'].fillna(0)
        quarter_data['injury_rate'] = (quarter_data['injury_count'] / quarter_data['total_plays']) * 1000
        
        # Free memory
        del injuries_with_quarter, quarter_injuries
        
        # 2. Down analysis
        logger.info("📊 Calculating injury rates by down...")
        # Get total plays by down
        down_stats = plays_data.groupby('down').size().reset_index(name='total_plays')
        
        # Get injuries by game and down
        injuries_with_down = injuries_data[['game_id']].merge(
            plays_data[['game_id', 'down']],
            on='game_id',
            how='left'
        )
        
        # Count injuries by down
        down_injuries = injuries_with_down.groupby('down').size().reset_index(name='injury_count')
        
        # Merge and calculate rates
        down_data = pd.merge(down_stats, down_injuries, on='down', how='left')
        down_data['injury_count'] = down_data['injury_count'].fillna(0)
        down_data['injury_rate'] = (down_data['injury_count'] / down_data['total_plays']) * 1000
        
        # Free memory
        del injuries_with_down, down_injuries
        
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
        
        # Get total plays by score differential bin
        score_stats = plays_data.groupby('score_differential_bin').size().reset_index(name='total_plays')
        
        # Get unique injuries by game and score differential
        injuries_with_score = injuries_data[['game_id']].merge(
            plays_data[['game_id', 'score_differential_bin']].drop_duplicates(),
            on='game_id',
            how='left'
        )
        
        # Count injuries by score differential bin
        score_injuries = injuries_with_score.groupby('score_differential_bin').size().reset_index(name='injury_count')
        
        # Merge and calculate rates
        score_data = pd.merge(score_stats, score_injuries, on='score_differential_bin', how='left')
        score_data['injury_count'] = score_data['injury_count'].fillna(0)
        score_data['injury_rate'] = (score_data['injury_count'] / score_data['total_plays']) * 1000
        
        # Free memory
        del injuries_with_score, score_injuries
        
        # 4. Time remaining analysis
        logger.info("📊 Calculating injury rates by time remaining...")
        # Create time remaining bins
        time_bins = [0, 900, 1800, 2700, 3600]  # 15-minute intervals
        plays_data['time_remaining_bin'] = pd.cut(
            plays_data['game_seconds_remaining'],
            bins=time_bins,
            labels=['0-15', '15-30', '30-45', '45-60']
        )
        
        # Get total plays by time remaining bin
        time_stats = plays_data.groupby('time_remaining_bin').size().reset_index(name='total_plays')
        
        # Get unique injuries by game and time remaining
        injuries_with_time = injuries_data[['game_id']].merge(
            plays_data[['game_id', 'time_remaining_bin']].drop_duplicates(),
            on='game_id',
            how='left'
        )
        
        # Count injuries by time remaining bin
        time_injuries = injuries_with_time.groupby('time_remaining_bin').size().reset_index(name='injury_count')
        
        # Merge and calculate rates
        time_data = pd.merge(time_stats, time_injuries, on='time_remaining_bin', how='left')
        time_data['injury_count'] = time_data['injury_count'].fillna(0)
        time_data['injury_rate'] = (time_data['injury_count'] / time_data['total_plays']) * 1000
        
        # Free memory
        del injuries_with_time, time_injuries
        
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
        trend_data = plays_data[['game_id']].copy()
        trend_data['time_index'] = trend_data['game_id'].str.replace('_', ' ')
        
        # Get total plays by time index
        trend_stats = trend_data.groupby('time_index').size().reset_index(name='total_plays')
        
        # Get injuries by time index
        trend_injuries = injuries_data.groupby('game_id').size().reset_index(name='injury_count')
        trend_injuries['time_index'] = trend_injuries['game_id'].str.replace('_', ' ')
        
        # Merge and calculate rates
        trend_data = pd.merge(trend_stats, trend_injuries[['time_index', 'injury_count']], on='time_index', how='left')
        trend_data['injury_count'] = trend_data['injury_count'].fillna(0)
        trend_data['injury_rate'] = (trend_data['injury_count'] / trend_data['total_plays']) * 1000
        
        # Sort by time index
        trend_data = trend_data.sort_values('time_index')
        
        # Free memory
        del trend_stats, trend_injuries
        
        # Create plot data for game situation plots
        plot_data = plays_data[['quarter', 'down', 'score_differential_bin', 'time_remaining_bin']].copy()
        
        # Merge with calculated rates
        plot_data = pd.merge(
            plot_data,
            quarter_data[['quarter', 'injury_rate']].rename(columns={'injury_rate': 'injury_rate_quarter'}),
            on='quarter',
            how='left'
        )
        plot_data = pd.merge(
            plot_data,
            down_data[['down', 'injury_rate']].rename(columns={'injury_rate': 'injury_rate_down'}),
            on='down',
            how='left'
        )
        plot_data = pd.merge(
            plot_data,
            score_data[['score_differential_bin', 'injury_rate']].rename(columns={'injury_rate': 'injury_rate_score'}),
            on='score_differential_bin',
            how='left'
        )
        plot_data = pd.merge(
            plot_data,
            time_data[['time_remaining_bin', 'injury_rate']].rename(columns={'injury_rate': 'injury_rate_time'}),
            on='time_remaining_bin',
            how='left'
        )
        
        # Add numeric columns for regression plots
        def convert_bin_to_numeric(bin_value):
            if pd.isna(bin_value) or str(bin_value).strip() == '':
                return np.nan
            try:
                if '-' in str(bin_value):
                    parts = str(bin_value).split('-')
                    if len(parts) == 2:
                        lower, upper = parts
                        if lower.strip() and upper.strip():
                            return (float(lower.strip()) + float(upper.strip())) / 2
            except (ValueError, IndexError):
                pass
            return np.nan

        plot_data['score_differential_numeric'] = plot_data['score_differential_bin'].apply(convert_bin_to_numeric)
        plot_data['time_remaining_numeric'] = plot_data['time_remaining_bin'].apply(convert_bin_to_numeric)
        
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