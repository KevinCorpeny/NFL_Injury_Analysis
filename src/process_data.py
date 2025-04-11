from pathlib import Path
from .config import Config
from .data.nfl_play_processor import NFLPlayProcessor
from .data.injury_processor import InjuryProcessor
from .utils.logging import logger

def main():
    """Main function to process NFL play and injury data."""
    try:
        # Load configuration
        config = Config.from_yaml("config/default.yaml")
        
        # Process NFL play data
        logger.info("Processing NFL play data")
        play_processor = NFLPlayProcessor(config)
        play_data = play_processor.process_data()
        
        # Save processed play data
        play_output_path = Path(config.data.processed_data_path) / "processed_plays.parquet"
        play_processor.save_data(play_data, play_output_path)
        
        # Process injury data
        logger.info("Processing injury data")
        injury_processor = InjuryProcessor(config)
        injury_data = injury_processor.process_data()
        
        # Save processed injury data
        injury_output_path = Path(config.data.processed_data_path) / "processed_injuries.parquet"
        injury_processor.save_data(injury_data, injury_output_path)
        
        # Log completion
        logger.info("Data processing completed successfully")
        
    except Exception as e:
        logger.error(f"Error in main processing: {str(e)}")
        raise

if __name__ == "__main__":
    main() 