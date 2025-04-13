from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import pandas as pd
import numpy as np
from src.data.base_processor import BaseProcessor
from src.config import Config
from src.utils.logging import logger

class InjuryProcessor(BaseProcessor):
    """Processor for NFL injury data.
    
    This class is responsible for loading, processing, and validating NFL injury data.
    """
    
    def __init__(self, config: Config):
        """Initialize the injury processor.
        
        Args:
            config: Configuration object with processing parameters
        """
        super().__init__(config)
        self.injury_features = config.model.features
        self.injury_data_path = Path(config.data.injury_data_path)
        self.processed_injury_path = Path(config.data.processed_data_path)
        self.injury_data = None
    
    def load_data(self) -> pd.DataFrame:
        """Load the injury data from CSV file.
        
        Returns:
            DataFrame with the loaded injury data
        """
        logger.info(f"Loading injury data from {self.injury_data_path}")
        
        if not self.injury_data_path.exists():
            raise FileNotFoundError(f"Injury data file not found at {self.injury_data_path}")
        
        self.injury_data = pd.read_csv(self.injury_data_path)
        logger.info(f"Loaded {len(self.injury_data)} injury records")
        
        # Log the columns available
        logger.info(f"Columns in the injury data: {self.injury_data.columns.tolist()}")
        
        return self.injury_data
    
    def process(self) -> pd.DataFrame:
        """Process the injury data.
        
        This method performs the following steps:
        1. Load the data if not already loaded
        2. Apply basic processing
        3. Handle missing values
        4. Add derived features
        5. Validate the processed data
        
        Returns:
            Processed injury data as a DataFrame
        """
        if self.injury_data is None:
            self.load_data()
            
        logger.info("Processing injury data...")
        
        # Apply basic processing
        self._standardize_column_names()
        self._convert_date_columns()
        
        # Handle missing values
        self._handle_missing_values()
        
        # Add derived features
        self._add_derived_features()
        
        # Validate the processed data
        self._validate_data()
        
        # Select relevant columns
        self._select_features()
        
        logger.info("Injury data processing completed")
        return self.injury_data
    
    # For backward compatibility with existing code
    def process_data(self) -> pd.DataFrame:
        """Process the injury data (alias for process method).
        
        Returns:
            Processed injury data as a DataFrame
        """
        return self.process()
    
    # For backward compatibility with existing code
    def save_data(self, data: pd.DataFrame, output_path: Union[str, Path]) -> None:
        """Save data to file (compatibility method).
        
        Args:
            data: DataFrame to save
            output_path: Path to save to
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save based on file extension
        if output_path.suffix.lower() == '.parquet':
            data.to_parquet(output_path, index=False)
        else:
            data.to_csv(output_path, index=False)
            
        logger.info(f"Data saved successfully to {output_path}")
        
    def _standardize_column_names(self) -> None:
        """Standardize column names to lowercase."""
        self.injury_data.columns = self.injury_data.columns.str.lower()
    
    def _convert_date_columns(self) -> None:
        """Convert date columns to datetime objects."""
        date_columns = [col for col in self.injury_data.columns if 'date' in col.lower()]
        
        for col in date_columns:
            if col in self.injury_data.columns:
                try:
                    self.injury_data[col] = pd.to_datetime(self.injury_data[col])
                    logger.info(f"Converted {col} to datetime")
                except Exception as e:
                    logger.warning(f"Could not convert {col} to datetime: {str(e)}")
    
    def _handle_missing_values(self) -> None:
        """Handle missing values in the injury data."""
        # Count missing values by column
        missing_values = self.injury_data.isna().sum()
        logger.info(f"Missing values before handling:\n{missing_values[missing_values > 0]}")
        
        # Fill missing values appropriately based on column type
        categorical_columns = self.injury_data.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            self.injury_data[col] = self.injury_data[col].fillna('Unknown')
        
        # Fill numeric columns with 0 or median based on context
        numeric_columns = self.injury_data.select_dtypes(include=['number']).columns
        for col in numeric_columns:
            if col.endswith('_id') or col.endswith('_count'):
                self.injury_data[col] = self.injury_data[col].fillna(0)
            else:
                self.injury_data[col] = self.injury_data[col].fillna(self.injury_data[col].median())
        
        # Log remaining missing values
        missing_after = self.injury_data.isna().sum()
        if missing_after.sum() > 0:
            logger.warning(f"Remaining missing values after handling:\n{missing_after[missing_after > 0]}")
    
    def _calculate_games_played(self) -> None:
        """Calculate the number of games played before injury for each player in each season.
        Accounts for bye weeks and different types of absences."""
        if not all(col in self.injury_data.columns for col in ['player_name', 'team', 'season', 'week']):
            logger.warning("Required columns missing for games played calculation")
            return

        # Sort data by player, season, and week
        self.injury_data = self.injury_data.sort_values(['player_name', 'team', 'season', 'week'])
        
        # Define types of absences that should not count as games played
        non_game_absences = [
            'Bye Week',
            'Illness',
            'Not Injury Related',
            'Personal',
            'Coach\'s Decision',
            'Not Listed'
        ]
        
        # Group by player, team, and season to calculate games played
        def calc_games_played(group):
            group = group.copy()
            
            # Initialize games played counter
            games_played = 0
            
            # Create a list to track games played for each week
            games_played_by_week = []
            
            # Iterate through weeks to calculate cumulative games played
            for week in range(1, 18):  # NFL regular season is 17 weeks
                if week < group['week'].min():
                    # Before first injury, count as games played
                    games_played += 1
                elif week == group['week'].min():
                    # At injury week, don't count as played
                    pass
                else:
                    # After injury, check if player was active
                    week_data = group[group['week'] == week]
                    if not week_data.empty:
                        status = week_data['game_status'].iloc[0]
                        practice_status = week_data['practice_status'].iloc[0]
                        
                        # Don't count if player was inactive for non-game reasons
                        if status in non_game_absences or practice_status in non_game_absences:
                            pass
                        else:
                            games_played += 1
                
                games_played_by_week.append(games_played)
            
            # Add the games played count to the injury record
            group['games_played_before_injury'] = games_played_by_week[group['week'].iloc[0] - 1]
            
            # Add additional tracking metrics
            group['total_season_games'] = games_played_by_week[-1]  # Total games played in season
            group['games_after_injury'] = group['total_season_games'] - group['games_played_before_injury']
            
            # Calculate injury timing metrics
            group['injury_week_percentage'] = group['week'] / 17  # Percentage through season
            group['games_played_percentage'] = group['games_played_before_injury'] / group['total_season_games']
            
            return group

        self.injury_data = self.injury_data.groupby(['player_name', 'team', 'season'], as_index=False).apply(calc_games_played)
        logger.info("Added enhanced games played calculation with bye week and absence tracking")

    def _add_derived_features(self) -> None:
        """Add derived features to the injury data."""
        # Calculate games played before injury with enhanced tracking
        self._calculate_games_played()

        # Add injury severity if not already present
        if 'injury_severity' not in self.injury_data.columns and 'game_status' in self.injury_data.columns:
            status_map = {
                'Out': 'Severe',
                'Doubtful': 'Moderate to Severe',
                'Questionable': 'Mild to Moderate',
                'Not Listed': 'None'
            }
            self.injury_data['injury_severity'] = self.injury_data['game_status'].map(status_map)
            logger.info("Added derived feature: injury_severity")
        
        # Add body region categorization if injury description is available
        if any(col for col in self.injury_data.columns if col in ['injury', 'injury_type', 'body_part']):
            injury_col = next((col for col in ['injury', 'injury_type', 'body_part'] 
                             if col in self.injury_data.columns), None)
            
            if injury_col:
                # Define region mappings
                lower_body = ['knee', 'ankle', 'foot', 'hip', 'thigh', 'hamstring', 'groin', 'leg', 'calf', 'quad']
                upper_body = ['shoulder', 'arm', 'elbow', 'hand', 'wrist', 'finger', 'back', 'chest', 'rib']
                head = ['head', 'concussion', 'neck', 'face', 'eye', 'jaw']
                
                # Function to categorize injuries
                def categorize_region(injury_text):
                    if pd.isna(injury_text) or injury_text == 'Unknown':
                        return 'Unknown'
                    
                    injury_text = str(injury_text).lower()
                    
                    if any(part in injury_text for part in lower_body):
                        return 'Lower Body'
                    elif any(part in injury_text for part in upper_body):
                        return 'Upper Body'
                    elif any(part in injury_text for part in head):
                        return 'Head/Neck'
                    else:
                        return 'Other'
                
                self.injury_data['body_region'] = self.injury_data[injury_col].apply(categorize_region)
                logger.info("Added derived feature: body_region")
        
        # Add season week formatting if needed
        if 'season' in self.injury_data.columns and 'week' in self.injury_data.columns:
            self.injury_data['season_week'] = self.injury_data['season'].astype(str) + '_' + self.injury_data['week'].astype(str)
            logger.info("Added derived feature: season_week")
    
    def _validate_data(self) -> None:
        """Validate the processed injury data."""
        # Check for required columns
        required_columns = ['season', 'week', 'player_name', 'position', 'team']
        missing_columns = [col for col in required_columns if col not in self.injury_data.columns]
        
        if missing_columns:
            logger.warning(f"Missing required columns in injury data: {missing_columns}")
        
        # Check for duplicate records
        duplicate_count = self.injury_data.duplicated().sum()
        if duplicate_count > 0:
            logger.warning(f"Found {duplicate_count} duplicate records in injury data")
            self.injury_data = self.injury_data.drop_duplicates()
            logger.info("Removed duplicate records")
    
    def _select_features(self) -> None:
        """Select relevant features based on the configuration."""
        if not self.injury_features:
            logger.info("No specific injury features selected, keeping all columns")
            return
        
        # Check if all requested features are available
        missing_features = [feat for feat in self.injury_features if feat not in self.injury_data.columns]
        if missing_features:
            logger.warning(f"Requested features not found in injury data: {missing_features}")
        
        # Select available features
        available_features = [feat for feat in self.injury_features if feat in self.injury_data.columns]
        if available_features:
            self.injury_data = self.injury_data[available_features]
            logger.info(f"Selected {len(available_features)} features from injury data")
    
    def merge_with_play_data(self, play_data: pd.DataFrame, on: List[str] = None) -> pd.DataFrame:
        """Merge injury data with play data.
        
        Args:
            play_data: DataFrame with NFL play data
            on: List of columns to merge on. If None, tries to infer common identifiers.
            
        Returns:
            Merged DataFrame with play and injury data
        """
        if self.injury_data is None:
            self.process()
        
        logger.info("Merging injury data with play data...")
        
        # If merge columns not specified, try to infer them
        if on is None:
            # Common identifiers between play and injury data
            potential_merge_columns = [
                ['season', 'week', 'team', 'player_id'],
                ['season', 'week', 'team', 'player_name'],
                ['season', 'week', 'team']
            ]
            
            # Find the first set of columns that exists in both DataFrames
            for columns in potential_merge_columns:
                if all(col in self.injury_data.columns for col in columns) and all(col in play_data.columns for col in columns):
                    on = columns
                    break
            
            if on is None:
                raise ValueError("Could not infer common columns to merge on. Please specify 'on' parameter.")
        
        logger.info(f"Merging on columns: {on}")
        
        # Check if merge columns exist in both DataFrames
        missing_in_injury = [col for col in on if col not in self.injury_data.columns]
        missing_in_play = [col for col in on if col not in play_data.columns]
        
        if missing_in_injury or missing_in_play:
            error_msg = []
            if missing_in_injury:
                error_msg.append(f"Columns missing in injury data: {missing_in_injury}")
            if missing_in_play:
                error_msg.append(f"Columns missing in play data: {missing_in_play}")
            raise ValueError("\n".join(error_msg))
        
        # Perform merge
        merged_data = pd.merge(
            play_data, 
            self.injury_data, 
            on=on,
            how='left',
            indicator=True,
            suffixes=('', '_injury')
        )
        
        # Log merge statistics
        match_count = (merged_data['_merge'] == 'both').sum()
        total_plays = len(play_data)
        total_injuries = len(self.injury_data)
        
        logger.info(f"Merge statistics:")
        logger.info(f"Total plays: {total_plays}")
        logger.info(f"Total injuries: {total_injuries}")
        logger.info(f"Plays with matching injury data: {match_count} ({match_count/total_plays:.2%})")
        
        # Remove merge indicator column
        merged_data = merged_data.drop(columns=['_merge'])
        
        return merged_data
    
    def save_processed_data(self, output_path: Optional[Union[str, Path]] = None) -> None:
        """Save the processed injury data to CSV.
        
        Args:
            output_path: Path to save the processed data. If None, use the default path.
        """
        if self.injury_data is None:
            raise ValueError("No data to save. Run process() first.")
        
        if output_path is None:
            output_dir = Path("data/processed")
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / "processed_injuries.csv"
        else:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.injury_data.to_csv(output_path, index=False)
        logger.info(f"Saved processed injury data to {output_path}") 