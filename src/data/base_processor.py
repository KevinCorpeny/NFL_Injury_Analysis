from typing import Dict, Any
import pandas as pd
from src.utils.logging import logger
from src.config import Config

class BaseProcessor:
    """Base class for data processors.
    
    This class provides common functionality for data processors.
    """
    
    def __init__(self, config: Config):
        """Initialize the base processor.
        
        Args:
            config: Configuration object with processing parameters
        """
        self.config = config
        
    def save_data(self, data: pd.DataFrame, output_path: str) -> None:
        """Save data to a file.
        
        Args:
            data: DataFrame to save
            output_path: Path to save the data to
        """
        try:
            # Determine file type from path
            if output_path.endswith('.csv'):
                data.to_csv(output_path, index=False)
            elif output_path.endswith('.parquet'):
                data.to_parquet(output_path, index=False)
            else:
                logger.warning(f"Unknown file extension for {output_path}, defaulting to CSV")
                data.to_csv(output_path, index=False)
                
            logger.info(f"Data saved successfully to {output_path}")
        except Exception as e:
            logger.error(f"Error saving data to {output_path}: {str(e)}")
            raise 