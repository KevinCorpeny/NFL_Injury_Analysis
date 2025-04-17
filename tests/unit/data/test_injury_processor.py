"""Unit tests for the InjuryProcessor class."""
import pytest
import pandas as pd
from src.data.injury_processor import InjuryProcessor

def test_injury_processor_initialization(mock_config):
    """Test that InjuryProcessor initializes correctly."""
    processor = InjuryProcessor(mock_config)
    assert processor.injury_features == mock_config.model.features
    assert processor.injury_data_path == mock_config.data.injury_data_path
    assert processor.processed_injury_path == mock_config.data.processed_data_path

def test_process_injuries(mock_config, sample_injuries_data):
    """Test that injuries are processed correctly."""
    processor = InjuryProcessor(mock_config)
    
    # Add required columns for validation
    sample_data = sample_injuries_data.copy()
    sample_data['player_name'] = ['Player 1', 'Player 2']
    sample_data['position'] = ['QB', 'WR']
    sample_data['team'] = ['Team A', 'Team B']
    
    # Set the data directly since we're not loading from file
    processor.injury_data = sample_data
    
    # Process the sample data
    processed_data = processor.process()
    
    # Check that the processed data has the expected columns
    expected_columns = ['season', 'week', 'game_id', 'injury_type', 'player_position']
    assert all(col in processed_data.columns for col in expected_columns)
    
    # Check that no data was lost during processing
    assert len(processed_data) == len(sample_data)
    
    # Check that data types are correct
    assert processed_data['season'].dtype == 'int64'
    assert processed_data['week'].dtype == 'int64'

def test_merge_with_plays(mock_config, sample_injuries_data, sample_plays_data):
    """Test that injuries merge correctly with play data."""
    processor = InjuryProcessor(mock_config)
    
    # Add required columns for merging
    injuries = sample_injuries_data.copy()
    injuries['team'] = ['Team A', 'Team B']
    plays = sample_plays_data.copy()
    plays['team'] = ['Team A'] * len(plays)
    
    # Set the injury data
    processor.injury_data = injuries
    
    # Merge injuries with plays
    merged_data = processor.merge_with_play_data(
        play_data=plays,
        on=['season', 'week', 'team']
    )
    
    # Check that the merge was successful
    assert len(merged_data) > 0
    
    # Check that we have both play and injury information
    expected_columns = [
        'season', 'week', 'game_id',  # Common columns
        'injury_type', 'player_position',  # Injury columns
        'quarter', 'down', 'play_type'  # Play columns
    ]
    assert all(col in merged_data.columns for col in expected_columns)

def test_validate_data(mock_config, sample_injuries_data):
    """Test that data validation works correctly."""
    processor = InjuryProcessor(mock_config)
    
    # Add required columns for validation
    valid_data = sample_injuries_data.copy()
    valid_data['player_name'] = ['Player 1', 'Player 2']
    valid_data['position'] = ['QB', 'WR']
    valid_data['team'] = ['Team A', 'Team B']
    
    # Set the data directly
    processor.injury_data = valid_data
    
    # Test with valid data
    processor._validate_data()  # Should not raise any exceptions
    
    # Test with invalid data (missing required column)
    invalid_data = valid_data.drop('player_name', axis=1)
    processor.injury_data = invalid_data
    with pytest.raises(ValueError, match="Missing required columns"):
        processor._validate_data()

def test_handle_missing_values(mock_config, sample_injuries_data):
    """Test that missing values are handled correctly."""
    processor = InjuryProcessor(mock_config)
    
    # Add some missing values
    data_with_missing = sample_injuries_data.copy()
    data_with_missing.loc[0, 'injury_type'] = None
    
    # Set the data directly
    processor.injury_data = data_with_missing
    
    # Handle missing values
    processor._handle_missing_values()
    
    # Check that missing values were handled
    assert not processor.injury_data['injury_type'].isna().any() 