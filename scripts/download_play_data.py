import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import nfl_data_py as nfl
import pandas as pd
import logging
from src.utils.logging import setup_logging

def main():
    # Set up logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Create directories if they don't exist
    data_dir = Path('data/nfl_plays')
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Define seasons to download (2018-2022)
    seasons = list(range(2018, 2023))
    
    try:
        logger.info(f"Downloading NFL play data for seasons {seasons}")
        
        # Download play-by-play data
        df = nfl.import_pbp_data(seasons)
        
        # Save to parquet file
        output_path = data_dir / 'plays.parquet'
        df.to_parquet(output_path)
        
        logger.info(f"Successfully downloaded and saved {len(df)} plays to {output_path}")
        logger.info(f"Data columns: {df.columns.tolist()}")
        
    except Exception as e:
        logger.error(f"Error downloading NFL play data: {str(e)}")
        raise

if __name__ == '__main__':
    main() 