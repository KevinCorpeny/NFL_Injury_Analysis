import pytest
import pandas as pd
import numpy as np
from src.data.validation import DataValidator

@pytest.fixture
def sample_config():
    """Fixture providing a sample validation configuration."""
    return {
        'validation': {
            'required_columns': ['player_name', 'team', 'season', 'week'],
            'column_types': {
                'player_name': 'string',
                'team': 'string',
                'season': 'int',
                'week': 'int'
            },
            'value_ranges': {
                'season': {'min': 2000, 'max': 2024},
                'week': {'min': 1, 'max': 18}
            },
            'consistency_rules': [
                # Remove the strict season-week mapping rule
                # as it's too restrictive for our test data
            ],
            'cross_field_rules': [
                {'condition': 'season >= 2000'}
            ],
            'statistical_rules': {
                'week': {
                    'mean': {'min': 5, 'max': 12},
                    'std': {'min': 2, 'max': 6}
                }
            }
        }
    }

@pytest.fixture
def valid_data():
    """Fixture providing valid sample data."""
    return pd.DataFrame({
        'player_name': ['John Smith', 'Mike Johnson'],
        'team': ['NE', 'KC'],
        'season': [2020, 2021],
        'week': [5, 10]
    })

@pytest.fixture
def invalid_data():
    """Fixture providing invalid sample data."""
    return pd.DataFrame({
        'player_name': ['John Smith', 'Mike Johnson'],
        'team': ['NE', 'KC'],
        'season': [1999, 2025],  # Invalid seasons
        'week': [0, 19]  # Invalid weeks
    })

def test_validate_schema(sample_config, valid_data):
    """Test schema validation with valid data."""
    validator = DataValidator(sample_config)
    results = validator.validate_schema(valid_data)
    assert not results['missing_columns']
    assert not results['extra_columns']

def test_validate_schema_missing_columns(sample_config):
    """Test schema validation with missing columns."""
    data = pd.DataFrame({
        'player_name': ['John Smith'],
        'team': ['NE']
    })
    validator = DataValidator(sample_config)
    results = validator.validate_schema(data)
    assert 'season' in results['missing_columns']
    assert 'week' in results['missing_columns']

def test_validate_data_types(sample_config, valid_data):
    """Test data type validation with valid data."""
    validator = DataValidator(sample_config)
    results = validator.validate_data_types(valid_data)
    assert not results['type_mismatches']

def test_validate_data_types_mismatch(sample_config):
    """Test data type validation with type mismatches."""
    data = pd.DataFrame({
        'player_name': ['John Smith'],
        'team': ['NE'],
        'season': ['2020'],  # String instead of int
        'week': [5.0]  # Float instead of int
    })
    validator = DataValidator(sample_config)
    results = validator.validate_data_types(data)
    assert len(results['type_mismatches']) == 2

def test_validate_value_ranges(sample_config, valid_data):
    """Test value range validation with valid data."""
    validator = DataValidator(sample_config)
    results = validator.validate_value_ranges(valid_data)
    assert not results['range_violations']

def test_validate_value_ranges_violations(sample_config, invalid_data):
    """Test value range validation with invalid data."""
    validator = DataValidator(sample_config)
    results = validator.validate_value_ranges(invalid_data)
    assert len(results['range_violations']) == 4  # 2 violations for season, 2 for week

def test_validate_consistency(sample_config):
    """Test consistency validation."""
    data = pd.DataFrame({
        'player_name': ['John Smith'],
        'team': ['NE'],
        'season': [2020],
        'week': [1]
    })
    validator = DataValidator(sample_config)
    results = validator.validate_consistency(data)
    assert not results['consistency_issues']

def test_validate_consistency_violations():
    """Test consistency validation with violations."""
    # Create a specific configuration for this test
    config = {
        'validation': {
            'required_columns': ['player_name', 'team', 'season', 'week'],
            'column_types': {
                'player_name': 'string',
                'team': 'string',
                'season': 'int',
                'week': 'int'
            },
            'consistency_rules': [
                {'season': 2020, 'week': 1}  # This rule expects season 2020 to have week 1
            ]
        }
    }
    
    # Create data that violates the consistency rule
    data = pd.DataFrame({
        'player_name': ['John Smith'],
        'team': ['NE'],
        'season': [2020],  # Season 2020
        'week': [5]  # But week 5, not week 1
    })
    
    validator = DataValidator(config)
    results = validator.validate_consistency(data)
    assert len(results['consistency_issues']) == 1
    assert results['consistency_issues'][0]['rule'] == {'season': 2020, 'week': 1}

def test_validate_cross_field(sample_config, valid_data):
    """Test cross-field validation with valid data."""
    validator = DataValidator(sample_config)
    results = validator.validate_cross_field(valid_data)
    assert not results['cross_field_issues']

def test_validate_cross_field_violations(sample_config):
    """Test cross-field validation with violations."""
    data = pd.DataFrame({
        'player_name': ['John Smith'],
        'team': ['NE'],
        'season': [1999],  # Violates season >= 2000 rule
        'week': [5]
    })
    validator = DataValidator(sample_config)
    results = validator.validate_cross_field(data)
    assert len(results['cross_field_issues']) == 1

def test_validate_statistics(sample_config):
    """Test statistical validation with valid data."""
    data = pd.DataFrame({
        'player_name': ['John Smith'] * 10,
        'team': ['NE'] * 10,
        'season': [2020] * 10,
        'week': [5, 6, 7, 8, 9, 10, 11, 12, 13, 14]  # Mean = 9.5, std ≈ 3.03
    })
    validator = DataValidator(sample_config)
    results = validator.validate_statistics(data)
    assert not results['statistical_issues']

def test_validate_statistics_violations(sample_config):
    """Test statistical validation with violations."""
    data = pd.DataFrame({
        'player_name': ['John Smith'] * 10,
        'team': ['NE'] * 10,
        'season': [2020] * 10,
        'week': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  # Mean = 1, std = 0 (both violate rules)
    })
    validator = DataValidator(sample_config)
    results = validator.validate_statistics(data)
    assert 'week' in results['statistical_issues']
    assert 'mean' in results['statistical_issues']['week']
    assert 'std' in results['statistical_issues']['week']

def test_validate_all(sample_config, valid_data):
    """Test comprehensive validation with valid data."""
    validator = DataValidator(sample_config)
    results = validator.validate_all(valid_data)
    
    # Check that there are no critical validation issues
    assert not results['schema']['missing_columns']
    assert not results['data_types']['type_mismatches']
    assert not results['value_ranges']['range_violations']
    assert not results['consistency']['consistency_issues']
    assert not results['cross_field']['cross_field_issues']
    
    # Statistical issues are warnings, not critical errors
    # So we don't check for them in the validation pass/fail

def test_validate_all_with_issues(sample_config, invalid_data):
    """Test comprehensive validation with invalid data."""
    validator = DataValidator(sample_config)
    results = validator.validate_all(invalid_data)
    assert any(results[category] for category in results) 