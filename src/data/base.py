from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional
import pandas as pd
from src.utils.logging import logger
from src.config import Config

class BaseDataProcessor(ABC):
    """Base class for data processing tasks."""
    
    def __init__(self, config: Config):
        self.config = config
        self.data: Optional[pd.DataFrame] = None
    
    @abstractmethod
    def load_data(self) -> pd.DataFrame:
        """Load data from source.
        
        Returns:
            DataFrame containing the loaded data
        """
        pass
    
    @abstractmethod
    def process_data(self) -> pd.DataFrame:
        """Process the loaded data.
        
        Returns:
            Processed DataFrame
        """
        pass
    
    def save_data(self, data: pd.DataFrame, filepath: Path) -> None:
        """Save processed data to file.
        
        Args:
            data: DataFrame to save
            filepath: Path to save the data to
        """
        try:
            filepath.parent.mkdir(parents=True, exist_ok=True)
            data.to_parquet(filepath)
            logger.info(f"Data saved successfully to {filepath}")
        except Exception as e:
            logger.error(f"Error saving data to {filepath}: {str(e)}")
            raise
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate the data meets basic quality requirements.
        
        Args:
            data: DataFrame to validate
        
        Returns:
            True if validation passes, False otherwise
        """
        try:
            # Check for empty DataFrame
            if data.empty:
                logger.error("Data validation failed: DataFrame is empty")
                return False
            
            # Check for missing values
            missing_stats = data.isnull().sum()
            if missing_stats.any():
                logger.warning(f"Missing values detected:\n{missing_stats[missing_stats > 0]}")
            
            # Add more validation as needed
            return True
            
        except Exception as e:
            logger.error(f"Error during data validation: {str(e)}")
            return False
    
    def get_data_stats(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Get basic statistics about the data.
        
        Args:
            data: DataFrame to analyze
        
        Returns:
            Dictionary containing data statistics
        """
        return {
            "row_count": len(data),
            "column_count": len(data.columns),
            "memory_usage": data.memory_usage(deep=True).sum(),
            "missing_values": data.isnull().sum().to_dict(),
            "dtypes": data.dtypes.to_dict()
        } 