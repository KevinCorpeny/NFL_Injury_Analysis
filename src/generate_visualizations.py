from pathlib import Path
import pandas as pd
from src.visualization.plotter import NFLPlotter
from src.utils.logging import logger

def main():
    """Generate all visualizations for the NFL injury analysis."""
    try:
        # Create output directory
        output_dir = Path("reports/figures")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize plotter
        plotter = NFLPlotter(output_dir)
        
        # Load processed data
        logger.info("Loading processed data")
        plays_path = Path("data/processed/processed_plays.parquet")
        injuries_path = Path("data/processed/processed_injuries.parquet")
        
        plays_data = pd.read_parquet(plays_path)
        injuries_data = pd.read_parquet(injuries_path)
        
        # Merge play and injury data
        data = pd.merge(
            plays_data,
            injuries_data,
            on=['game_id', 'play_id'],
            how='left'
        )
        
        # Add injury occurrence flag
        data['injury_occurred'] = data['injury_type'].notna()
        
        # Generate all visualizations
        logger.info("Generating visualizations")
        
        # Basic injury analysis
        plotter.plot_injury_by_play_type(data)
        plotter.plot_injury_by_position(data)
        plotter.plot_injury_severity_distribution(data)
        plotter.plot_injury_trend_over_time(data)
        
        # Advanced analysis
        plotter.plot_injury_correlation_heatmap(data)
        plotter.plot_injury_by_game_situation(data)
        plotter.plot_injury_by_weather(data)
        
        logger.info("All visualizations generated successfully")
        
    except Exception as e:
        logger.error(f"Error generating visualizations: {str(e)}")
        raise

if __name__ == "__main__":
    main() 