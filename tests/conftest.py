"""Common test fixtures and configuration."""
import os
import pytest
import pandas as pd
from pathlib import Path
from types import SimpleNamespace

@pytest.fixture
def mock_config(project_root):
    """Create a mock configuration for testing."""
    # Create nested SimpleNamespace objects to mock the config structure
    model_config = SimpleNamespace(
        features=['season', 'week', 'game_id', 'injury_type', 'player_position']
    )
    
    data_config = SimpleNamespace(
        injury_data_path=project_root / 'data/injury_reports/raw_injuries.csv',
        processed_data_path=project_root / 'data/processed/processed_injuries.parquet'
    )
    
    return SimpleNamespace(
        model=model_config,
        data=data_config
    )

@pytest.fixture
def test_data_dir():
    """Return the path to the test data directory."""
    return Path(__file__).parent / 'data'

@pytest.fixture
def sample_plays_data():
    """Create a sample plays DataFrame for testing."""
    return pd.DataFrame({
        'season': [2022, 2022, 2022, 2022],
        'week': [1, 1, 1, 1],
        'quarter': [1, 2, 3, 4],
        'down': [1, 2, 3, 1],
        'game_id': ['2022_01', '2022_01', '2022_01', '2022_01'],
        'play_type': ['run', 'pass', 'run', 'pass'],
        'score_differential': [0, 7, -7, 14],
        'time_remaining': [900, 600, 300, 100]
    })

@pytest.fixture
def sample_injuries_data():
    """Create a sample injuries DataFrame for testing."""
    return pd.DataFrame({
        'season': [2022, 2022],
        'week': [1, 1],
        'game_id': ['2022_01', '2022_01'],
        'injury_type': ['knee', 'ankle'],
        'player_position': ['QB', 'WR']
    })

@pytest.fixture
def project_root():
    """Return the project root directory."""
    return Path(__file__).parent.parent 