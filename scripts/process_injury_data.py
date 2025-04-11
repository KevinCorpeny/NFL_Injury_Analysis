import sys
from pathlib import Path
import yaml

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.data.injury_processor import InjuryProcessor
from src.utils.logging import logger

def main():
    """Process NFL injury data using the InjuryProcessor."""
    try:
        # Load configuration
        config_path = Path("config/default.yaml")
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        
        logger.info("Starting injury data processing")
        
        # Create and run the injury data processor
        processor = InjuryProcessor(config)
        processed_data = processor.process()
        
        # Save the processed data
        processor.save_processed_data()
        
        # Print some statistics
        logger.info(f"\nProcessed data statistics:")
        logger.info(f"Total records: {len(processed_data)}")
        
        # Show value counts for key columns
        for col in ['position', 'team', 'game_status', 'body_region', 'injury_severity']:
            if col in processed_data.columns:
                logger.info(f"\n{col.title()} distribution:")
                logger.info(processed_data[col].value_counts().head(10))
        
        logger.info("Injury data processing completed successfully")
        
    except Exception as e:
        logger.error(f"Error processing injury data: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main() 