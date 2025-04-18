#!/usr/bin/env python3
"""
Enhanced Risk Factor Identification Module

This module implements advanced statistical and machine learning methods to identify
key risk factors for NFL injuries, with improved accuracy and robustness.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_curve, roc_auc_score
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import statsmodels.api as sm
from typing import Dict, List, Tuple, Optional
import logging
import joblib
from pathlib import Path
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import xgboost as xgb
import lightgbm as lgb
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.model_selection import ParameterGrid
import time
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedRiskFactorAnalyzer:
    """Enhanced class for identifying and analyzing injury risk factors."""
    
    def __init__(self, data_dir: str = "data/processed"):
        """Initialize the enhanced risk factor analyzer."""
        self.data_dir = Path(data_dir)
        self.model = None
        self.scaler = None
        self.feature_importance = None
        self.risk_factors = None
        self.best_params = None
        self.feature_selector = None
        self.feature_columns = None
        self.feature_names = None
        self.analysis_results = {}
    
    def set_feature_columns(self, columns: List[str]) -> None:
        """Set the feature columns to use for analysis."""
        self.feature_columns = columns
        logger.info(f"Set feature columns: {columns}")
    
    def load_data(self, filename: str = "processed_injuries.parquet") -> pd.DataFrame:
        """Load and prepare the injury data with enhanced validation."""
        try:
            data_path = self.data_dir / filename
            if filename.endswith('.parquet'):
                df = pd.read_parquet(data_path)
            else:
                df = pd.read_csv(data_path)
            logger.info(f"Loaded data from {data_path}")
            
            # Enhanced data validation
            self._validate_data(df)
            
            # Set default feature columns if not already set
            if self.feature_columns is None:
                # Exclude identifier and target columns
                exclude_columns = ['gsis_id', 'player_name', 'first_name', 'last_name', 
                                 'injury_type', 'report_secondary_injury', 'game_status',
                                 'practice_primary_injury', 'practice_secondary_injury',
                                 'practice_status', 'date_modified']
                self.feature_columns = [col for col in df.columns if col not in exclude_columns]
                logger.info(f"Automatically set feature columns: {self.feature_columns}")
            
            return df
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def _validate_data(self, df: pd.DataFrame) -> None:
        """Enhanced data validation."""
        required_columns = ['season', 'week', 'team', 'position', 'injury_type']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Check for data quality issues
        self._check_data_quality(df)
    
    def _check_data_quality(self, df: pd.DataFrame) -> None:
        """Check data quality and log issues."""
        # Check for missing values
        missing_values = df.isna().sum()
        if missing_values.any():
            logger.warning(f"Missing values found:\n{missing_values[missing_values > 0]}")
        
        # Check for outliers in numerical columns
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
        for col in numerical_cols:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            outliers = df[(df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)][col]
            if len(outliers) > 0:
                logger.warning(f"Outliers found in {col}: {len(outliers)} values")
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features for model training."""
        logger.info("Preparing features...")
        
        # Create target variable
        y = self._create_target_variable(df)
        
        # Define meaningful feature groups
        player_features = ['position', 'games_played_before_injury', 'total_season_games', 
                         'games_played_percentage']
        game_context_features = ['game_type', 'quarter', 'down', 'score_differential_bin', 
                               'time_remaining_bin']
        injury_features = ['injury_severity', 'body_region', 'injury_week_percentage']
        
        # Combine all relevant features
        self.feature_columns = player_features + game_context_features + injury_features
        
        # Select features
        X = df[self.feature_columns].copy()
        
        # Handle missing values
        X = self._handle_missing_values(X)
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Log feature information
        logger.info(f"Selected features ({len(self.feature_names)}):")
        for feature in self.feature_names:
            logger.info(f"  - {feature}")
        
        return X, y
    
    def _create_target_variable(self, df: pd.DataFrame) -> pd.Series:
        """Create enhanced target variable."""
        # Define what counts as a significant injury
        significant_injuries = [
            'ACL', 'MCL', 'Ankle', 'Concussion', 'Hamstring', 'Knee',
            'Shoulder', 'Groin', 'Hip', 'Foot', 'Back', 'Neck'
        ]
        
        # Consider injury severity and type
        injury_conditions = (
            # Game injuries
            ((df['injury_type'].notna()) & 
             (df['injury_type'].str.contains('|'.join(significant_injuries), case=False, na=False)) &
             ~df['injury_type'].isin(['Not Listed', 'Not Injury Related'])) |
            # Practice injuries
            ((df['practice_primary_injury'].notna()) & 
             (df['practice_primary_injury'].str.contains('|'.join(significant_injuries), case=False, na=False)) &
             ~df['practice_primary_injury'].isin(['Not Listed', 'Not Injury Related']))
        )
        
        return injury_conditions.astype(int)
    
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create enhanced features."""
        # Time-based features
        df['season_progress'] = df['week'] / 17  # Normalized season progress
        df['is_playoff_week'] = (df['week'] > 17).astype(int)
        
        # Player workload features
        if 'games_played_before_injury' in df.columns:
            df['workload_ratio'] = df['games_played_before_injury'] / df['total_season_games']
        
        # Team performance features
        df = self._add_team_performance_features(df)
        
        # Position-specific features
        df = self._add_position_features(df)
        
        return df
    
    def _add_team_performance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add team performance metrics."""
        # Calculate team win rates
        team_stats = df.groupby(['team', 'season']).agg({
            'is_injured': 'mean',
            'week': 'count'
        }).reset_index()
        team_stats.columns = ['team', 'season', 'team_injury_rate', 'games_played']
        
        # Merge back to main dataframe
        df = df.merge(team_stats, on=['team', 'season'], how='left')
        return df
    
    def _add_position_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add position-specific features."""
        # Calculate position-specific injury rates
        position_stats = df.groupby('position').agg({
            'is_injured': 'mean',
            'week': 'count'
        }).reset_index()
        position_stats.columns = ['position', 'position_injury_rate', 'position_games']
        
        # Merge back to main dataframe
        df = df.merge(position_stats, on='position', how='left')
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values with advanced imputation."""
        # Separate numerical and categorical columns
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        # Use KNN imputation for numerical columns
        if len(numerical_cols) > 0:
            imputer = KNNImputer(n_neighbors=5)
            df[numerical_cols] = imputer.fit_transform(df[numerical_cols])
        
        # Use mode imputation for categorical columns
        for col in categorical_cols:
            df[col] = df[col].fillna(df[col].mode()[0])
        
        return df
    
    def _prepare_final_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare final feature matrix and target."""
        # Drop identifier and post-injury columns
        columns_to_drop = [
            'gsis_id', 'player_name', 'first_name', 'last_name',
            'injury_type', 'report_secondary_injury', 'game_status',
            'practice_primary_injury', 'practice_secondary_injury',
            'practice_status', 'date_modified'
        ]
        df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
        
        # Separate features and target
        X = df.drop(['is_injured'], axis=1)
        y = df['is_injured']
        
        # Create preprocessing pipeline
        numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
        categorical_cols = X.select_dtypes(include=['object']).columns
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', RobustScaler(), numerical_cols),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
            ]
        )
        
        # Apply preprocessing and convert to dense array
        X_processed = preprocessor.fit_transform(X)
        if hasattr(X_processed, 'toarray'):
            X_processed = X_processed.toarray()
        
        return pd.DataFrame(X_processed), y
    
    def train_model(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train the risk factor analysis model."""
        logger.info("Training model...")
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Log target distribution
        logger.info("Target distribution in training set:")
        logger.info(y_train.value_counts())
        logger.info("Target distribution in test set:")
        logger.info(y_test.value_counts())
        
        # Define numerical and categorical features
        numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Create preprocessing pipeline
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ]
        )
        
        # Create model pipeline with SMOTE
        pipeline = ImbPipeline([
            ('preprocessor', preprocessor),
            ('smote', SMOTE(random_state=42)),
            ('clf', RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            ))
        ])
        
        # Fit the model
        pipeline.fit(X_train, y_train)
        
        # Store the model
        self.model = pipeline
        
        # Get feature names after preprocessing
        feature_names = []
        for name, transformer, features in preprocessor.transformers_:
            if name == 'cat':
                # Get feature names from OneHotEncoder
                cat_feature_names = transformer.get_feature_names_out(features)
                feature_names.extend(cat_feature_names)
            else:
                feature_names.extend(features)
        
        self.feature_names = feature_names
        logger.info(f"Final feature names after preprocessing: {len(feature_names)}")
        
        # Evaluate the model
        y_pred = pipeline.predict(X_test)
        y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc_roc = roc_auc_score(y_test, y_pred_proba)
        
        # Log evaluation metrics
        logger.info("\nEnhanced Model Evaluation:")
        logger.info(f"Accuracy: {accuracy:.3f}")
        logger.info(f"Precision: {precision:.3f}")
        logger.info(f"Recall: {recall:.3f}")
        logger.info(f"F1 Score: {f1:.3f}")
        logger.info(f"AUC-ROC: {auc_roc:.3f}")
        
        # Log classification report
        logger.info("\nClassification Report:")
        logger.info(classification_report(y_test, y_pred))
        
        # Analyze risk factors
        analysis_results = self.analyze_risk_factors(X, y)
        
        # Save model and parameters
        self.save_model({
            'model_type': 'RandomForestClassifier',
            'feature_names': feature_names,
            'analysis_results': analysis_results,
            'metrics': {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'auc_roc': auc_roc
            }
        })
    
    def identify_risk_factors(self) -> List[str]:
        """Identify the most important risk factors from the trained model."""
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call train_model() first.")
        
        # Get feature importances from the trained model
        rf_model = self.model.named_steps['clf']
        feature_importances = rf_model.feature_importances_
        
        # Ensure we have the same number of features and importances
        min_length = min(len(self.feature_names), len(feature_importances))
        feature_names = self.feature_names[:min_length]
        importances = feature_importances[:min_length]
        
        # Create a DataFrame of feature importances
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        # Get top 10 risk factors
        top_risk_factors = importance_df.head(10)['feature'].tolist()
        
        # Log the top risk factors
        logger.info("\nTop Risk Factors:")
        for i, factor in enumerate(top_risk_factors, 1):
            importance = importance_df.loc[importance_df['feature'] == factor, 'importance'].iloc[0]
            logger.info(f"{i}. {factor} (importance: {importance:.4f})")
        
        return top_risk_factors
    
    def save_model(self, params: Dict = None, output_dir: str = "models") -> None:
        """Save the trained model and related artifacts."""
        if self.model is None:
            raise ValueError("No model to save. Train a model first.")
        
        # Create output directory if it doesn't exist
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save the model
        model_path = output_path / "enhanced_risk_factor_model.joblib"
        joblib.dump(self.model, model_path)
        logger.info(f"Saved model to {model_path}")
        
        # Save parameters if provided
        if params is not None:
            params_path = output_path / "model_params.json"
            with open(params_path, 'w') as f:
                json.dump(params, f, indent=4)
            logger.info(f"Saved model parameters to {params_path}")

    def analyze_risk_factors(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Dict[str, float]]:
        """Analyze risk factors by calculating odds ratios and relative risks."""
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call train_model() first.")
        
        # Get feature importances from the trained model
        rf_model = self.model.named_steps['clf']
        feature_importances = rf_model.feature_importances_
        
        # Get preprocessor
        preprocessor = self.model.named_steps['preprocessor']
        
        # Get numerical and categorical features
        num_features = X.select_dtypes(include=['int64', 'float64']).columns
        cat_features = X.select_dtypes(include=['object', 'category']).columns
        
        # Create DataFrame with features and target
        df = X.copy()
        df['is_injured'] = y
        
        # Calculate statistics for each feature
        results = {}
        
        # Process numerical features
        for feature in num_features:
            try:
                # Calculate median for binary split
                median = df[feature].median()
                high_risk = df[feature] > median
                
                # Create contingency table
                contingency = pd.crosstab(high_risk, df['is_injured'])
                
                # Calculate odds ratio
                odds_ratio = (contingency.iloc[1, 1] * contingency.iloc[0, 0]) / \
                            (contingency.iloc[1, 0] * contingency.iloc[0, 1])
                
                # Calculate relative risk
                risk_exposed = contingency.iloc[1, 1] / (contingency.iloc[1, 0] + contingency.iloc[1, 1])
                risk_unexposed = contingency.iloc[0, 1] / (contingency.iloc[0, 0] + contingency.iloc[0, 1])
                relative_risk = risk_exposed / risk_unexposed if risk_unexposed > 0 else float('inf')
                
                # Calculate prevalence
                prevalence = high_risk.mean()
                
                # Get feature importance
                importance = feature_importances[list(self.feature_names).index(feature)]
                
                results[feature] = {
                    'odds_ratio': odds_ratio,
                    'relative_risk': relative_risk,
                    'prevalence': prevalence,
                    'importance': importance
                }
            except Exception as e:
                logger.warning(f"Could not analyze numerical feature {feature}: {str(e)}")
                continue
        
        # Process categorical features
        for feature in cat_features:
            # Get all categories
            categories = df[feature].unique()
            
            for category in categories:
                try:
                    # Create binary indicator
                    is_category = df[feature] == category
                    
                    # Create contingency table
                    contingency = pd.crosstab(is_category, df['is_injured'])
                    
                    # Calculate odds ratio
                    odds_ratio = (contingency.iloc[1, 1] * contingency.iloc[0, 0]) / \
                                (contingency.iloc[1, 0] * contingency.iloc[0, 1])
                    
                    # Calculate relative risk
                    risk_exposed = contingency.iloc[1, 1] / (contingency.iloc[1, 0] + contingency.iloc[1, 1])
                    risk_unexposed = contingency.iloc[0, 1] / (contingency.iloc[0, 0] + contingency.iloc[0, 1])
                    relative_risk = risk_exposed / risk_unexposed if risk_unexposed > 0 else float('inf')
                    
                    # Calculate prevalence
                    prevalence = is_category.mean()
                    
                    # Get feature importance
                    feature_name = f"{feature}_{category}"
                    importance = feature_importances[list(self.feature_names).index(feature_name)]
                    
                    results[feature_name] = {
                        'odds_ratio': odds_ratio,
                        'relative_risk': relative_risk,
                        'prevalence': prevalence,
                        'importance': importance
                    }
                except Exception as e:
                    logger.warning(f"Could not analyze categorical feature {feature}_{category}: {str(e)}")
                    continue
        
        # Sort by importance
        sorted_results = dict(sorted(
            results.items(),
            key=lambda x: x[1]['importance'],
            reverse=True
        ))
        
        # Log the analysis results
        logger.info("\nRisk Factor Analysis:")
        for feature, stats in sorted_results.items():
            logger.info(f"\n{feature}:")
            logger.info(f"  Odds Ratio: {stats['odds_ratio']:.2f}")
            logger.info(f"  Relative Risk: {stats['relative_risk']:.2f}")
            logger.info(f"  Prevalence: {stats['prevalence']:.2%}")
            logger.info(f"  Importance: {stats['importance']:.4f}")
        
        self.analysis_results = sorted_results
        return sorted_results

def main():
    """Main function to demonstrate usage."""
    analyzer = EnhancedRiskFactorAnalyzer()
    
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