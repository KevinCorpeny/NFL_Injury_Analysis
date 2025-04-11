from pathlib import Path
from typing import Dict, List
import pandas as pd
import nfl_data_py as nfl
from src.data.base import BaseDataProcessor
from src.config import Config
from src.utils.logging import logger

class NFLPlayProcessor(BaseDataProcessor):
    """Processor for NFL play-by-play data."""
    
    def __init__(self, config: Config):
        super().__init__(config)
        self.seasons = config.data.seasons
    
    def load_data(self) -> pd.DataFrame:
        """Load NFL play-by-play data for specified seasons.
        
        Returns:
            DataFrame containing play-by-play data
        """
        logger.info(f"Loading NFL play-by-play data for seasons {self.seasons}")
        try:
            # Load play-by-play data using nfl_data_py
            df = nfl.import_pbp_data(self.seasons)
            logger.info(f"Successfully loaded {len(df)} plays")
            
            # Log available columns
            logger.info("Available columns in play-by-play data:")
            for col in sorted(df.columns):
                logger.info(f"- {col}")
            
            return df
        except Exception as e:
            logger.error(f"Error loading NFL play data: {str(e)}")
            raise
    
    def process_data(self) -> pd.DataFrame:
        """Process NFL play-by-play data.
        
        Returns:
            Processed DataFrame with relevant features
        """
        if self.data is None:
            self.data = self.load_data()
        
        logger.info("Processing NFL play data")
        try:
            # Get available features
            available_features = [f for f in self.config.model.features if f in self.data.columns]
            missing_features = [f for f in self.config.model.features if f not in self.data.columns]
            
            if missing_features:
                logger.warning(f"Some features are not available in the data: {missing_features}")
                logger.info(f"Using available features: {available_features}")
            
            # Select relevant columns
            df = self.data[available_features].copy()
            
            # Check for duplicate columns
            duplicate_cols = df.columns[df.columns.duplicated()].tolist()
            if duplicate_cols:
                logger.warning(f"Found duplicate columns: {duplicate_cols}")
                # Remove duplicate columns, keeping the first occurrence
                df = df.loc[:, ~df.columns.duplicated()]
            
            # Add derived features
            df = self._add_derived_features(df)
            
            # Handle missing values
            df = self._handle_missing_values(df)
            
            # Validate processed data
            if not self.validate_data(df):
                raise ValueError("Data validation failed after processing")
            
            logger.info(f"Successfully processed {len(df)} plays")
            return df
            
        except Exception as e:
            logger.error(f"Error processing NFL play data: {str(e)}")
            raise
    
    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived features to the dataset.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with additional features
        """
        # Add quarter information
        if 'game_seconds_remaining' in df.columns:
            df['quarter'] = df['game_seconds_remaining'].apply(
                lambda x: 1 if x > 2700 else (2 if x > 1800 else (3 if x > 900 else 4))
            )
            
            # Add time remaining in quarter
            df['quarter_seconds_remaining'] = df['game_seconds_remaining'] % 900
        
        # Add score differential category if score_differential exists
        if 'score_differential' in df.columns:
            df['score_differential_category'] = pd.cut(
                df['score_differential'],
                bins=[-float('inf'), -14, -7, 0, 7, 14, float('inf')],
                labels=['>14 behind', '7-14 behind', '0-7 behind', '0-7 ahead', '7-14 ahead', '>14 ahead']
            )
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with handled missing values
        """
        # Log missing values before handling
        missing_before = df.isnull().sum()
        if missing_before.any():
            logger.warning(f"Missing values before handling:\n{missing_before[missing_before > 0]}")
        
        # Handle missing values based on column type
        for col in df.columns:
            col_series = df[col]
            if pd.api.types.is_numeric_dtype(col_series):
                # For numeric columns, fill with median
                df[col] = col_series.fillna(col_series.median())
            elif pd.api.types.is_string_dtype(col_series) or pd.api.types.is_categorical_dtype(col_series):
                # For string/categorical columns, fill with mode
                df[col] = col_series.fillna(col_series.mode()[0] if not col_series.mode().empty else 'unknown')
            elif pd.api.types.is_datetime64_any_dtype(col_series):
                # For datetime columns, fill with a default date or leave as NaT
                pass
        
        # Log missing values after handling
        missing_after = df.isnull().sum()
        if missing_after.any():
            logger.error(f"Missing values remain after handling:\n{missing_after[missing_after > 0]}")
        
        return df 