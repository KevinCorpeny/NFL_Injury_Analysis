from typing import Dict, List, Any, Optional, Union
import pandas as pd
import numpy as np
from datetime import datetime
from src.utils.logging import logger
from src.config import ValidationConfig

class DataValidator:
    """Class for comprehensive data validation of NFL injury data."""
    
    def __init__(self, config: ValidationConfig):
        self.config = config
        self.validation_results = {}
        
    def validate_schema(self, data: pd.DataFrame) -> Dict[str, List[str]]:
        """Check required columns and if configured, allow columns
        """
        results = {
            'missing_columns': [],
            'extra_columns': []
        }

        required = self.config.required_columns
        allowed = self.config.allowed_columns

        results['missing_columns'] = [col for col in required if col not in data.columns]

        if allowed:
            #only flag extras if a full allowed-columns list was configured
            results['extra_columns'] = [col for col in data.columns if col not in allowed]

        if results['missing_columns']:
            logger.error(f"Missing columns: {results['missing_columns']}")

        if results['extra_columns']:
            logger.warning(f"Extra columns: {results['extra_columns']}")

        return results
    
    def validate_data_types(self, data: pd.DataFrame) -> Dict[str, List[str]]:
        """Validate data types of columns.
        
        Args:
            data: DataFrame to validate
            
        Returns:
            Dictionary with validation results
        """
        results = {'type_mismatches': []}
        type_rules = self.validation_rules.get('column_types', {})
        
        for col, expected_type in type_rules.items():
            if col not in data.columns:
                continue
                
            actual_type = str(data[col].dtype)
            if not self._check_type_match(actual_type, expected_type):
                results['type_mismatches'].append({
                    'column': col,
                    'expected': expected_type,
                    'actual': actual_type
                })
                logger.error(f"Type mismatch in column {col}: expected {expected_type}, got {actual_type}")
                
        return results
    
    def validate_value_ranges(self, data: pd.DataFrame) -> Dict[str, List[Dict]]:
        """Validate that values fall within expected ranges.
        
        Args:
            data: DataFrame to validate
            
        Returns:
            Dictionary with validation results
        """
        results = {'range_violations': []}
        range_rules = self.validation_rules.get('value_ranges', {})
        
        for col, rules in range_rules.items():
            if col not in data.columns:
                continue
                
            if 'min' in rules and data[col].min() < rules['min']:
                results['range_violations'].append({
                    'column': col,
                    'issue': 'below_minimum',
                    'value': data[col].min(),
                    'threshold': rules['min']
                })
                
            if 'max' in rules and data[col].max() > rules['max']:
                results['range_violations'].append({
                    'column': col,
                    'issue': 'above_maximum',
                    'value': data[col].max(),
                    'threshold': rules['max']
                })
                
        return results
    
    def validate_consistency(self, data: pd.DataFrame) -> Dict[str, List[Dict]]:
        """Validate consistency between related fields.
        
        Args:
            data: DataFrame to validate
            
        Returns:
            Dictionary with validation results
        """
        results = {'consistency_issues': []}
        consistency_rules = self.validation_rules.get('consistency_rules', [])
        
        for rule in consistency_rules:
            if not self._check_consistency_rule(data, rule):
                results['consistency_issues'].append({
                    'rule': rule,
                    'description': f"Consistency rule violated: {rule}"
                })
                
        return results
    
    def validate_cross_field(self, data: pd.DataFrame) -> Dict[str, List[Dict]]:
        """Validate relationships between different fields.
        
        Args:
            data: DataFrame to validate
            
        Returns:
            Dictionary with validation results
        """
        results = {'cross_field_issues': []}
        cross_field_rules = self.validation_rules.get('cross_field_rules', [])
        
        for rule in cross_field_rules:
            if not self._check_cross_field_rule(data, rule):
                results['cross_field_issues'].append({
                    'rule': rule,
                    'description': f"Cross-field rule violated: {rule}"
                })
                
        return results
    
    def validate_statistics(self, data: pd.DataFrame) -> Dict[str, Dict]:
        """Validate statistical properties of the data.
        
        Args:
            data: DataFrame to validate
            
        Returns:
            Dictionary with validation results
        """
        results = {'statistical_issues': {}}
        stat_rules = self.validation_rules.get('statistical_rules', {})
        
        for col, rules in stat_rules.items():
            if col not in data.columns:
                continue
                
            col_stats = {}
            if 'mean' in rules:
                actual_mean = data[col].mean()
                if not self._check_value_within_range(actual_mean, rules['mean']):
                    col_stats['mean'] = {
                        'actual': actual_mean,
                        'expected': rules['mean']
                    }
                    
            if 'std' in rules:
                actual_std = data[col].std()
                if not self._check_value_within_range(actual_std, rules['std']):
                    col_stats['std'] = {
                        'actual': actual_std,
                        'expected': rules['std']
                    }
                    
            if col_stats:
                results['statistical_issues'][col] = col_stats
                
        return results
    
    def validate_all(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Run all validation checks on the data.
        
        Args:
            data: DataFrame to validate
            
        Returns:
            Dictionary with all validation results
        """
        self.validation_results = {
            'schema': self.validate_schema(data),
            'data_types': self.validate_data_types(data),
            'value_ranges': self.validate_value_ranges(data),
            'consistency': self.validate_consistency(data),
            'cross_field': self.validate_cross_field(data),
            'statistics': self.validate_statistics(data)
        }
        
        # Check if any validation failed
        has_errors = False
        
        # Check for missing columns
        if self.validation_results['schema']['missing_columns']:
            has_errors = True
            
        # Check for type mismatches
        if self.validation_results['data_types']['type_mismatches']:
            has_errors = True
            
        # Check for range violations
        if self.validation_results['value_ranges']['range_violations']:
            has_errors = True
            
        # Check for consistency issues
        if self.validation_results['consistency']['consistency_issues']:
            has_errors = True
            
        # Check for cross-field issues
        if self.validation_results['cross_field']['cross_field_issues']:
            has_errors = True
            
        # Statistical issues are warnings, not errors
        if self.validation_results['statistics']['statistical_issues']:
            logger.warning("Statistical anomalies detected in the data")
        
        if has_errors:
            logger.warning("Data validation completed with issues")
        else:
            logger.info("Data validation completed successfully")
            
        return self.validation_results
    
    def _check_type_match(self, actual_type: str, expected_type: str) -> bool:
        """Check if actual data type matches expected type.
        
        Args:
            actual_type: Actual data type
            expected_type: Expected data type
            
        Returns:
            True if types match, False otherwise
        """
        type_mapping = {
            'int': ['int64', 'int32', 'int16', 'int8'],
            'float': ['float64', 'float32', 'float16'],
            'string': ['object', 'string'],
            'datetime': ['datetime64[ns]']
        }
        
        if expected_type in type_mapping:
            return actual_type in type_mapping[expected_type]
        return actual_type == expected_type
    
    def _check_consistency_rule(self, data: pd.DataFrame, rule: Dict) -> bool:
        """Check a consistency rule.
        
        Args:
            data: DataFrame to check
            rule: Consistency rule to check
            
        Returns:
            True if rule is satisfied, False otherwise
        """
        # Example rule: {'field1': 'value1', 'field2': 'value2'}
        for field, value in rule.items():
            if field not in data.columns:
                return False
            # Use .all() to check if all values in the Series match
            if not (data[field] == value).all():
                return False
        return True
    
    def _check_cross_field_rule(self, data: pd.DataFrame, rule: Dict) -> bool:
        """Check a cross-field rule.
        
        Args:
            data: DataFrame to check
            rule: Cross-field rule to check
            
        Returns:
            True if rule is satisfied, False otherwise
        """
        # Example rule: {'condition': 'field1 > field2'}
        condition = rule['condition']
        try:
            # Evaluate the condition for each row and check if all rows satisfy it
            return eval(condition, {}, data).all()
        except Exception as e:
            logger.error(f"Error evaluating cross-field rule '{condition}': {str(e)}")
            return False
    
    def _check_value_within_range(self, value: float, range_spec: Dict) -> bool:
        """Check if a value is within specified range.
        
        Args:
            value: Value to check
            range_spec: Range specification with min and max
            
        Returns:
            True if value is within range, False otherwise
        """
        if 'min' in range_spec and value < range_spec['min']:
            return False
        if 'max' in range_spec and value > range_spec['max']:
            return False
        return True 