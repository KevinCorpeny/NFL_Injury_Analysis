#!/usr/bin/env python3
"""
Risk Factor Identification Module

This module implements statistical and machine learning methods to identify
key risk factors for NFL injuries.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_curve, roc_auc_score
import statsmodels.api as sm
from typing import Dict, List, Tuple, Optional
import logging
import joblib
from pathlib import Path
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RiskFactorAnalyzer:
    """Class for identifying and analyzing injury risk factors."""
    
    def __init__(self, data_dir: str = "data/processed"):
        """Initialize the risk factor analyzer."""
        self.data_dir = Path(data_dir)
        self.model = None
        self.scaler = StandardScaler()
        self.feature_importance = None
        self.risk_factors = None
    
    def load_data(self, filename: str = "processed_injuries.parquet") -> pd.DataFrame:
        """Load and prepare the injury data."""
        try:
            data_path = self.data_dir / filename
            if filename.endswith('.parquet'):
                df = pd.read_parquet(data_path)
            else:
                df = pd.read_csv(data_path)
            logger.info(f"Loaded data from {data_path}")
            logger.info(f"Columns in data: {df.columns.tolist()}")
            
            # Log initial injury counts
            logger.info(f"Total rows: {len(df)}")
            
            # Check for empty strings or 'nan' strings
            for col in ['injury_type', 'report_secondary_injury', 'practice_primary_injury', 'practice_secondary_injury']:
                if df[col].dtype == 'object':
                    # Log unique values before cleaning
                    logger.info(f"\nUnique values in {col} before cleaning:")
                    logger.info(df[col].value_counts().head())
                    
                    # Clean the data
                    df[col] = df[col].replace('', np.nan)
                    df[col] = df[col].replace('nan', np.nan)
                    df[col] = df[col].replace('None', np.nan)
                    df[col] = df[col].replace('null', np.nan)
                    df[col] = df[col].replace('NULL', np.nan)
                    df[col] = df[col].replace('Unknown', np.nan)  # Treat 'Unknown' as missing
                    if col == 'practice_primary_injury':
                        df[col] = df[col].replace('Not Injury Related', np.nan)  # Treat 'Not Injury Related' as missing
                    if col == 'practice_secondary_injury':
                        df[col] = df[col].replace('No Secondary Injury', np.nan)  # Treat 'No Secondary Injury' as missing
                    
                    # Log unique values after cleaning
                    logger.info(f"\nUnique values in {col} after cleaning:")
                    logger.info(df[col].value_counts().head())
            
            # Log corrected injury counts
            logger.info("\nCorrected injury counts:")
            logger.info(f"Rows with injury_type: {df['injury_type'].notna().sum()}")
            logger.info(f"Rows with report_secondary_injury: {df['report_secondary_injury'].notna().sum()}")
            logger.info(f"Rows with practice_primary_injury: {df['practice_primary_injury'].notna().sum()}")
            logger.info(f"Rows with practice_secondary_injury: {df['practice_secondary_injury'].notna().sum()}")
            
            return df
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features for analysis.
        
        Args:
            df: Raw data DataFrame
            
        Returns:
            Tuple of (features, target)
        """
        logger.info(f"Initial data shape: {df.shape}")
        
        # Create binary target: 1 if injured, 0 if not
        # Consider both game and practice injuries
        df['is_injured'] = 0  # Initialize all as not injured
        
        # Mark rows with any type of injury
        injury_conditions = (
            df['injury_type'].notna() |  # Game injuries
            df['practice_primary_injury'].notna() |  # Practice injuries
            df['practice_secondary_injury'].notna()  # Secondary practice injuries
        )
        
        # Log the conditions
        logger.info("\nInjury conditions breakdown:")
        logger.info(f"Rows with injury_type: {df['injury_type'].notna().sum()}")
        logger.info(f"Rows with practice_primary_injury: {df['practice_primary_injury'].notna().sum()}")
        logger.info(f"Rows with practice_secondary_injury: {df['practice_secondary_injury'].notna().sum()}")
        logger.info(f"Rows meeting any injury condition: {injury_conditions.sum()}")
        
        df.loc[injury_conditions, 'is_injured'] = 1
        logger.info(f"Target distribution:\n{df['is_injured'].value_counts(normalize=True)}")
        
        # Drop columns that are either identifiers or post-injury features
        columns_to_drop = [
            'gsis_id', 'player_name', 'first_name', 'last_name',
            'injury_type', 'report_secondary_injury', 'game_status',
            'practice_primary_injury', 'practice_secondary_injury',
            'practice_status', 'date_modified', 'injury_severity',
            'body_region', 'games_after_injury', 'games_played_before_injury',
            'total_season_games', 'injury_week_percentage', 'games_played_percentage'
        ]
        df = df.drop(columns=columns_to_drop)
        logger.info(f"Shape after dropping columns: {df.shape}")
        
        # Convert time remaining bins to numerical values
        if 'time_remaining_bin' in df.columns:
            time_bin_mapping = {
                '0-15 min': 7.5,
                '15-30 min': 22.5,
                '30-45 min': 37.5,
                '45-60 min': 52.5,
                '60+ min': 60
            }
            df['time_remaining_minutes'] = df['time_remaining_bin'].map(time_bin_mapping)
            df = df.drop('time_remaining_bin', axis=1)
            logger.info("Converted time remaining bins to minutes")
        
        # Convert score differential bins to numerical values
        if 'score_differential_bin' in df.columns:
            score_bin_mapping = {
                'Leading by 14+': 14,
                'Leading by 7-13': 10,
                'Leading by 1-6': 3,
                'Tied': 0,
                'Trailing by 1-6': -3,
                'Trailing by 7-13': -10,
                'Trailing by 14+': -14
            }
            # Convert to string type first to handle any categorical issues
            df['score_differential_bin'] = df['score_differential_bin'].astype(str)
            # Replace any invalid values with 'Tied'
            valid_bins = set(score_bin_mapping.keys())
            df.loc[~df['score_differential_bin'].isin(valid_bins), 'score_differential_bin'] = 'Tied'
            # Map to numerical values
            df['score_differential_numeric'] = df['score_differential_bin'].map(score_bin_mapping)
            df = df.drop('score_differential_bin', axis=1)
            logger.info("Converted score differential bins to numeric values")
        
        # Handle missing values
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
        numerical_cols = [col for col in numerical_cols if col != 'is_injured']  # Don't impute target
        missing_values = df[numerical_cols].isna().sum()
        if missing_values.any():
            logger.info(f"Missing values before imputation:\n{missing_values[missing_values > 0]}")
            for col in numerical_cols:
                df[col] = df[col].fillna(df[col].median())
            logger.info("Imputed missing values with median")
        
        # Identify numerical and categorical columns
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
        numerical_cols = [col for col in numerical_cols if col != 'is_injured']  # Exclude target
        categorical_cols = df.select_dtypes(include=['object']).columns
        logger.info(f"Numerical columns: {len(numerical_cols)}")
        logger.info(f"Categorical columns: {len(categorical_cols)}")
        
        # Scale numerical features (excluding target)
        if len(numerical_cols) > 0:
            df[numerical_cols] = self.scaler.fit_transform(df[numerical_cols])
            logger.info("Scaled numerical features")
        
        # Convert categorical variables to dummy variables
        if len(categorical_cols) > 0:
            df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
            logger.info(f"Created dummy variables. New shape: {df.shape}")
        
        # Separate features and target
        X = df.drop(['is_injured'], axis=1)
        y = df['is_injured']
        logger.info(f"Final feature matrix shape: {X.shape}")
        logger.info(f"Target distribution:\n{y.value_counts(normalize=True)}")
        
        return X, y
    
    def train_model(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Train the risk factor model.
        
        Args:
            X: Feature matrix
            y: Target variable
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Calculate class weights
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
        
        # Initialize model with balanced class weights
        self.model = RandomForestClassifier(
            n_estimators=500,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight=class_weight_dict,
            random_state=42
        )
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Calculate optimal threshold using ROC curve
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        
        # Adjust predictions using optimal threshold
        y_pred_adj = (y_pred_proba >= optimal_threshold).astype(int)
        
        # Evaluate model
        logger.info("\nModel Evaluation:")
        logger.info(f"Accuracy: {accuracy_score(y_test, y_pred_adj):.2f}")
        logger.info(f"Precision: {precision_score(y_test, y_pred_adj):.2f}")
        logger.info(f"Recall: {recall_score(y_test, y_pred_adj):.2f}")
        logger.info(f"F1 Score: {f1_score(y_test, y_pred_adj):.2f}")
        logger.info(f"ROC AUC: {roc_auc_score(y_test, y_pred_proba):.2f}")
        
        # Print classification report
        logger.info("\nClassification Report:")
        logger.info(classification_report(y_test, y_pred_adj))
        
        # Plot feature importance
        self.plot_feature_importance(X.columns)
        
        # Store optimal threshold
        self.optimal_threshold = optimal_threshold
        
        # Store test data for later use
        self.X_test = X_test
        self.y_test = y_test
    
    def plot_feature_importance(self, feature_names: pd.Index) -> None:
        """Plot feature importance."""
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(12, 8))
        plt.title('Feature Importance')
        plt.bar(range(len(importances)), importances[indices])
        plt.xticks(range(len(importances)), feature_names[indices], rotation=90)
        plt.tight_layout()
        
        # Save plot
        plot_path = self.data_dir / 'feature_importance.png'
        plt.savefig(plot_path)
        plt.close()
        
        logger.info(f"Feature importance plot saved to {plot_path}")
        
    def identify_risk_factors(self, threshold: float = 0.01) -> List[str]:
        """
        Identify significant risk factors.
        
        Args:
            threshold: Minimum importance threshold
            
        Returns:
            List of significant risk factors
        """
        if self.model is None:
            raise ValueError("Model must be trained first")
            
        # Get feature importance
        importances = self.model.feature_importances_
        feature_names = self.X_test.columns
        
        # Identify significant factors
        self.risk_factors = [
            feature_names[i] for i in range(len(importances))
            if importances[i] > threshold
        ]
        
        # Log top 10 factors
        logger.info("\nTop 10 Risk Factors:")
        sorted_indices = np.argsort(importances)[::-1]
        for i in range(min(10, len(sorted_indices))):
            idx = sorted_indices[i]
            logger.info(f"{feature_names[idx]}: {importances[idx]:.4f}")
        
        return self.risk_factors
    
    def analyze_risk_factors(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """
        Perform detailed analysis of identified risk factors.
        
        Args:
            X: Prepared feature matrix
            y: Target variable
            
        Returns:
            Dictionary containing risk factor analysis results
        """
        if self.risk_factors is None:
            raise ValueError("Risk factors must be identified first")
        
        analysis_results = {}
        
        for factor in self.risk_factors:
            # Calculate odds ratios
            contingency_table = pd.crosstab(X[factor], y)
            odds_ratio = (contingency_table.iloc[1,1] * contingency_table.iloc[0,0]) / \
                        (contingency_table.iloc[1,0] * contingency_table.iloc[0,1])
            
            # Calculate relative risk
            risk_in_exposed = contingency_table.iloc[1,1] / contingency_table.iloc[1].sum()
            risk_in_unexposed = contingency_table.iloc[0,1] / contingency_table.iloc[0].sum()
            relative_risk = risk_in_exposed / risk_in_unexposed
            
            analysis_results[factor] = {
                'odds_ratio': odds_ratio,
                'relative_risk': relative_risk,
                'prevalence': X[factor].mean()
            }
        
        return analysis_results
    
    def save_model(self, output_dir: str = "models") -> None:
        """Save the trained model and analysis results."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        joblib.dump(self.model, output_path / 'risk_factor_model.joblib')
        joblib.dump(self.scaler, output_path / 'scaler.joblib')
        
        # Save feature importance
        if self.feature_importance is not None:
            self.feature_importance.to_csv(
                output_path / 'feature_importance.csv',
                index=False
            )
        
        logger.info(f"Saved model and analysis results to {output_path}")

def main():
    """Main function to demonstrate usage."""
    analyzer = RiskFactorAnalyzer()
    
    # Load and prepare data
    df = analyzer.load_data()
    X, y = analyzer.prepare_features(df)
    
    # Train model and identify risk factors
    analyzer.train_model(X, y)
    risk_factors = analyzer.identify_risk_factors()
    
    # Analyze risk factors
    analysis_results = analyzer.analyze_risk_factors(X, y)
    
    # Save results
    analyzer.save_model()
    
    # Print results
    print("\nIdentified Risk Factors:")
    for factor in risk_factors:
        print(f"\n{factor}:")
        print(f"  Odds Ratio: {analysis_results[factor]['odds_ratio']:.2f}")
        print(f"  Relative Risk: {analysis_results[factor]['relative_risk']:.2f}")
        print(f"  Prevalence: {analysis_results[factor]['prevalence']:.2%}")

if __name__ == "__main__":
    main() 