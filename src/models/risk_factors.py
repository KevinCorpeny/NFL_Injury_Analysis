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
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedRiskFactorAnalyzer:
    """Enhanced class for identifying and analyzing severe injury risk factors.
    
    IMPORTANT: This implementation focuses specifically on SEVERE injuries,
    defined as injuries that resulted in a player being ruled 'Out' for a game.
    Minor injuries, questionable statuses, and non-injury absences are not
    considered positive cases in this model.
    
    This design choice allows the model to:
    - Focus on injuries with significant impact on player availability
    - Reduce noise from minor/day-to-day injuries
    - Create more reliable and actionable predictions
    - Target prevention of serious injuries specifically
    
    The model achieves ~17% positive case rate, representing severe injuries
    that caused players to miss games entirely.
    """
    
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
    
    def _create_target_variable(self, df: pd.DataFrame) -> pd.Series:
        """Create target variable using only severe injury occurrence information.
        
        IMPORTANT NOTE:
        This implementation specifically focuses on SEVERE injuries only, defined as:
        - Injuries that resulted in 'Out' game status
        - Excludes minor injuries, questionable statuses, and non-injury absences
        - ~17% of the dataset represents severe injuries
        
        This is a deliberate choice to:
        1. Focus on injuries with significant impact on player availability
        2. Reduce noise from minor/day-to-day injuries
        3. Create a more reliable prediction target
        
        Args:
            df: Input DataFrame
            
        Returns:
            Series indicating severe injury occurrence (1) or no severe injury (0)
        """
        # Only consider severe injuries (game status = 'Out')
        # This is a strict definition focusing on injuries that caused players to miss games
        injury_conditions = (
            (df['injury_type'].notna()) & 
            (df['game_status'] == 'Out') &  # Only severe injuries resulting in missed games
            ~df['injury_type'].isin(['Not Listed', 'Not Injury Related', 'Rest'])
        )
        
        # Log injury distribution
        injury_counts = injury_conditions.value_counts()
        injury_percentages = injury_conditions.value_counts(normalize=True) * 100
        
        logger.info("\nInjury Distribution (Severe Injuries Only):")
        logger.info(f"No Severe Injury: {injury_counts[0]} ({injury_percentages[0]:.1f}%)")
        logger.info(f"Severe Injury: {injury_counts[1]} ({injury_percentages[1]:.1f}%)")
        logger.info("\nNote: This model focuses only on severe injuries (game status = 'Out')")
        logger.info("Minor injuries and questionable statuses are not considered positive cases")
        
        return injury_conditions.astype(int)
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features for model training, ensuring no data leakage.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (feature matrix, target variable)
        """
        logger.info("Preparing features...")
        
        # Create target variable first
        y = self._create_target_variable(df)
        
        # Define pre-injury feature groups (no injury-specific information)
        identifier_features = [
            'player_name'  # Keep for splitting, will be removed before training
        ]
        
        player_features = [
            'position',
            'games_played_before_injury',
            'total_season_games',
            'games_played_percentage'
        ]
        
        game_context_features = [
            'game_type',
            'quarter',
            'down',
            'score_differential_bin',
            'time_remaining_bin'
        ]
        
        # Combine all relevant features
        self.feature_columns = identifier_features + player_features + game_context_features
        
        # Select features
        X = df[self.feature_columns].copy()
        
        # Handle missing values
        X = self._handle_missing_values(X)
        
        # Store feature names (excluding identifiers)
        self.feature_names = player_features + game_context_features
        
        # Log feature information
        logger.info(f"Selected features ({len(self.feature_names)}):")
        for feature in self.feature_names:
            logger.info(f"  - {feature}")
        
        return X, y
    
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create enhanced features with interactions and temporal patterns."""
        # Time-based features
        df['season_progress'] = df['week'] / 17  # Normalized season progress
        df['is_playoff_week'] = (df['week'] > 17).astype(int)
        
        # Player workload features
        if 'games_played_before_injury' in df.columns:
            df['workload_ratio'] = df['games_played_before_injury'] / df['total_season_games']
            df['recent_workload'] = df['games_played_before_injury'].rolling(window=3, min_periods=1).mean()
        
        # Team performance features
        df = self._add_team_performance_features(df)
        
        # Position-specific features
        df = self._add_position_features(df)
        
        # Position-game context interactions
        df = self._add_position_game_interactions(df)
        
        # Temporal injury patterns
        df = self._add_temporal_features(df)
        
        return df

    def _add_position_game_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add interaction features between position and game context."""
        # Position-quarter interactions
        df['position_quarter'] = df['position'] + '_q' + df['quarter'].astype(str)
        
        # Position-score differential interactions
        df['position_score_diff'] = df['position'] + '_' + df['score_differential_bin']
        
        # Position-time remaining interactions
        df['position_time_remaining'] = df['position'] + '_' + df['time_remaining_bin']
        
        # Position-down interactions
        df['position_down'] = df['position'] + '_d' + df['down'].astype(str)
        
        # Position-specific workload features
        position_workload = df.groupby(['position', 'week']).agg({
            'games_played_before_injury': 'mean',
            'total_season_games': 'mean'
        }).reset_index()
        position_workload['position_workload_ratio'] = position_workload['games_played_before_injury'] / position_workload['total_season_games']
        df = df.merge(position_workload[['position', 'week', 'position_workload_ratio']], 
                     on=['position', 'week'], how='left')
        
        return df

    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add temporal features for injury patterns."""
        # Sort by player and week
        df = df.sort_values(['player_name', 'week'])
        
        # Previous injury history
        df['previous_injuries'] = df.groupby('player_name')['is_injured'].shift(1).fillna(0)
        df['injury_streak'] = df.groupby('player_name')['is_injured'].transform(
            lambda x: x.rolling(window=3, min_periods=1).sum()
        )
        
        # Time since last injury
        df['weeks_since_last_injury'] = df.groupby('player_name').apply(
            lambda x: x['week'] - x[x['is_injured'] == 1]['week'].shift(1)
        ).reset_index(level=0, drop=True).fillna(999)  # Large number for players with no previous injuries
        
        # Rolling injury rates by position
        position_injury_rates = df.groupby(['position', 'week']).agg({
            'is_injured': 'mean'
        }).reset_index()
        position_injury_rates['rolling_injury_rate'] = position_injury_rates.groupby('position')['is_injured'].transform(
            lambda x: x.rolling(window=3, min_periods=1).mean()
        )
        df = df.merge(position_injury_rates[['position', 'week', 'rolling_injury_rate']], 
                     on=['position', 'week'], how='left')
        
        return df

    def _add_team_performance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add team performance metrics."""
        # Calculate team win rates and injury rates
        team_stats = df.groupby(['team', 'season']).agg({
            'is_injured': 'mean',
            'week': 'count'
        }).reset_index()
        team_stats.columns = ['team', 'season', 'team_injury_rate', 'games_played']
        
        # Calculate team performance metrics
        team_performance = df.groupby(['team', 'week']).agg({
            'score_differential': 'mean',
            'quarter': 'mean'
        }).reset_index()
        team_performance['team_performance'] = team_performance['score_differential'].rolling(window=3, min_periods=1).mean()
        
        # Merge back to main dataframe
        df = df.merge(team_stats, on=['team', 'season'], how='left')
        df = df.merge(team_performance[['team', 'week', 'team_performance']], 
                     on=['team', 'week'], how='left')
        
        return df

    def _add_position_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add position-specific features."""
        # Calculate position-specific injury rates
        position_stats = df.groupby('position').agg({
            'is_injured': 'mean',
            'week': 'count',
            'games_played_before_injury': 'mean',
            'total_season_games': 'mean'
        }).reset_index()
        position_stats.columns = ['position', 'position_injury_rate', 'position_games', 
                                'position_avg_games_played', 'position_avg_total_games']
        
        # Calculate position-specific workload
        position_stats['position_workload'] = position_stats['position_avg_games_played'] / position_stats['position_avg_total_games']
        
        # Merge back to main dataframe
        df = df.merge(position_stats, on='position', how='left')
        
        # Add position group features
        df['position_group'] = df['position'].map({
            'QB': 'QB',
            'RB': 'RB',
            'WR': 'WR',
            'TE': 'TE',
            'T': 'OL',
            'G': 'OL',
            'C': 'OL',
            'DE': 'DL',
            'DT': 'DL',
            'LB': 'LB',
            'CB': 'DB',
            'S': 'DB',
            'K': 'ST',
            'P': 'ST',
            'LS': 'ST'
        })
        
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
    
    def _perform_cross_validation(self, X: pd.DataFrame, y: pd.Series, train_players: np.ndarray) -> Dict:
        """Perform enhanced cross-validation with player-level splitting.
        
        Args:
            X: Feature matrix with player_name column
            y: Target variable
            train_players: Array of player names for training set
            
        Returns:
            Dictionary containing cross-validation results
        """
        from sklearn.model_selection import StratifiedKFold
        from sklearn.metrics import make_scorer, f1_score, roc_auc_score, precision_score, recall_score
        
        # Create custom scorers
        scoring = {
            'f1': make_scorer(f1_score),
            'roc_auc': make_scorer(roc_auc_score),
            'precision': make_scorer(precision_score),
            'recall': make_scorer(recall_score)
        }
        
        # Create player-level stratified k-fold
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        # Store results for each fold
        results = {
            'metrics': {
                'train': {},
                'test': {}
            },
            'fold_details': []
        }
        
        # Get player-level injury status for stratification
        player_injury_status = pd.DataFrame({
            'player_name': train_players,
            'has_severe_injury': [y[X['player_name'] == player].any() for player in train_players]
        })
        
        logger.info("\nPlayer-Level Injury Distribution:")
        logger.info(f"Players with severe injuries: {player_injury_status['has_severe_injury'].sum()} ({player_injury_status['has_severe_injury'].mean()*100:.1f}%)")
        logger.info(f"Players without severe injuries: {len(player_injury_status) - player_injury_status['has_severe_injury'].sum()} ({100 - player_injury_status['has_severe_injury'].mean()*100:.1f}%)")
        
        # Perform cross-validation
        for fold, (train_idx, test_idx) in enumerate(cv.split(
            player_injury_status,
            player_injury_status['has_severe_injury']
        )):
            # Get player names for this fold
            fold_train_players = player_injury_status.iloc[train_idx]['player_name']
            fold_test_players = player_injury_status.iloc[test_idx]['player_name']
            
            # Get indices for all observations of these players
            train_mask = X['player_name'].isin(fold_train_players)
            test_mask = X['player_name'].isin(fold_test_players)
            
            # Split data
            X_train_fold = X[train_mask].copy()
            X_test_fold = X[test_mask].copy()
            y_train_fold = y[train_mask]
            y_test_fold = y[test_mask]
            
            # Remove player_name before training (but keep the original DataFrames for future folds)
            X_train_fold_features = X_train_fold.drop('player_name', axis=1)
            X_test_fold_features = X_test_fold.drop('player_name', axis=1)
            
            # Log fold details
            logger.info(f"\nFold {fold + 1} Details:")
            logger.info(f"Training Players: {len(fold_train_players)}")
            logger.info(f"Test Players: {len(fold_test_players)}")
            logger.info(f"Training Observations: {len(X_train_fold)}")
            logger.info(f"Test Observations: {len(X_test_fold)}")
            logger.info(f"Training Injury Rate: {y_train_fold.mean():.2%}")
            logger.info(f"Test Injury Rate: {y_test_fold.mean():.2%}")
            
            # Fit model
            self.model.fit(X_train_fold_features, y_train_fold)
            
            # Get predictions
            y_train_pred = self.model.predict(X_train_fold_features)
            y_test_pred = self.model.predict(X_test_fold_features)
            y_test_proba = self.model.predict_proba(X_test_fold_features)[:, 1]
            
            # Calculate metrics
            fold_metrics = {
                'train': {
                    'f1': f1_score(y_train_fold, y_train_pred),
                    'roc_auc': roc_auc_score(y_train_fold, y_train_pred),
                    'precision': precision_score(y_train_fold, y_train_pred),
                    'recall': recall_score(y_train_fold, y_train_pred)
                },
                'test': {
                    'f1': f1_score(y_test_fold, y_test_pred),
                    'roc_auc': roc_auc_score(y_test_fold, y_test_proba),
                    'precision': precision_score(y_test_fold, y_test_pred),
                    'recall': recall_score(y_test_fold, y_test_pred)
                }
            }
            
            # Store fold details
            fold_result = {
                'fold': fold + 1,
                'train_players': len(fold_train_players),
                'test_players': len(fold_test_players),
                'train_observations': len(X_train_fold),
                'test_observations': len(X_test_fold),
                'train_injury_rate': y_train_fold.mean(),
                'test_injury_rate': y_test_fold.mean()
            }
            results['fold_details'].append(fold_result)
            
            # Update metrics
            for metric in scoring.keys():
                if metric not in results['metrics']['train']:
                    results['metrics']['train'][metric] = []
                    results['metrics']['test'][metric] = []
                results['metrics']['train'][metric].append(fold_metrics['train'][metric])
                results['metrics']['test'][metric].append(fold_metrics['test'][metric])
        
        # Calculate mean and std of metrics
        for metric in scoring.keys():
            results['metrics']['train'][metric] = {
                'mean': np.mean(results['metrics']['train'][metric]),
                'std': np.std(results['metrics']['train'][metric])
            }
            results['metrics']['test'][metric] = {
                'mean': np.mean(results['metrics']['test'][metric]),
                'std': np.std(results['metrics']['test'][metric])
            }
        
        # Log results
        logger.info("\nCross-Validation Results:")
        for metric in scoring.keys():
            logger.info(f"\n{metric.upper()}:")
            logger.info(f"  Train: {results['metrics']['train'][metric]['mean']:.3f} ± {results['metrics']['train'][metric]['std']:.3f}")
            logger.info(f"  Test:  {results['metrics']['test'][metric]['mean']:.3f} ± {results['metrics']['test'][metric]['std']:.3f}")
        
        return results
    
    def _plot_cv_results(self, cv_results: Dict, scoring: Dict) -> None:
        """Plot cross-validation results.
        
        Args:
            cv_results: Dictionary of cross-validation results
            scoring: Dictionary of scoring metrics
        """
        # Set style
        plt.style.use('seaborn-v0_8')  # Use a valid style name
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        # Plot each metric
        for i, (metric, ax) in enumerate(zip(scoring.keys(), axes)):
            train_scores = cv_results[f'train_{metric}']
            test_scores = cv_results[f'test_{metric}']
            
            # Create box plot
            sns.boxplot(data=[train_scores, test_scores], ax=ax)
            ax.set_xticklabels(['Train', 'Test'])
            ax.set_title(f'{metric.upper()} Scores')
            ax.set_ylabel('Score')
        
        plt.tight_layout()
        
        # Save plot
        output_dir = Path("reports/figures")
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / "cross_validation_results.png")
        plt.close()
        
        logger.info(f"Saved cross-validation results plot to {output_dir / 'cross_validation_results.png'}")

    def train_model(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train the model with enhanced cross-validation.
        
        Args:
            X: Feature matrix with player_name column
            y: Target variable
        """
        from sklearn.model_selection import train_test_split, GridSearchCV
        from sklearn.preprocessing import StandardScaler, OneHotEncoder
        from sklearn.compose import ColumnTransformer
        from imblearn.pipeline import Pipeline as ImbPipeline
        from imblearn.over_sampling import SMOTE
        import xgboost as xgb
        
        # Split data into training and test sets at player level
        unique_players = X['player_name'].unique()
        train_players, test_players = train_test_split(
            unique_players,
            test_size=0.2,
            random_state=42,
            stratify=[y[X['player_name'] == player].any() for player in unique_players]
        )
        
        # Create masks for training and test sets
        train_mask = X['player_name'].isin(train_players)
        test_mask = X['player_name'].isin(test_players)
        
        # Split the data
        X_train = X[train_mask].copy()
        X_test = X[test_mask].copy()
        y_train = y[train_mask]
        y_test = y[test_mask]
        
        # Create feature matrices without player_name
        X_train_features = X_train.drop('player_name', axis=1)
        X_test_features = X_test.drop('player_name', axis=1)
        
        # Initialize the model
        numerical_features = X_train_features.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = X_train_features.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Create preprocessing pipeline
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ]
        )
        
        # Calculate class weights
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        weight_dict = {i: w for i, w in enumerate(class_weights)}
        
        # Define XGBoost model with class weights
        xgb_model = xgb.XGBClassifier(
            objective='binary:logistic',
            scale_pos_weight=weight_dict[1]/weight_dict[0],  # Handle class imbalance
            random_state=42,
            n_jobs=-1
        )
        
        # Create model pipeline with SMOTE
        self.model = ImbPipeline([
            ('preprocessor', preprocessor),
            ('smote', SMOTE(random_state=42)),
            ('clf', xgb_model)
        ])
        
        # Define hyperparameter grid
        param_grid = {
            'clf__n_estimators': [100, 200, 300],
            'clf__max_depth': [3, 5, 7],
            'clf__learning_rate': [0.01, 0.1, 0.2],
            'clf__subsample': [0.8, 0.9, 1.0],
            'clf__colsample_bytree': [0.8, 0.9, 1.0],
            'clf__min_child_weight': [1, 3, 5]
        }
        
        # Perform grid search with cross-validation
        grid_search = GridSearchCV(
            self.model,
            param_grid,
            cv=5,
            scoring='f1',
            n_jobs=-1,
            verbose=1
        )
        
        # Log target distribution
        logger.info("Target distribution in training set:")
        logger.info(y_train.value_counts())
        logger.info("Target distribution in test set:")
        logger.info(y_test.value_counts())
        
        # Fit grid search
        logger.info("Performing hyperparameter tuning...")
        grid_search.fit(X_train_features, y_train)
        
        # Store best model and parameters
        self.model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        
        logger.info(f"Best parameters: {self.best_params}")
        
        # Perform cross-validation
        cv_results = self._perform_cross_validation(X_train, y_train, train_players)
        
        # Store results
        self.cv_results = cv_results
        
        # Store feature names
        self.feature_names = X_train_features.columns.tolist()
        
        # Evaluate on test set
        test_predictions = self.model.predict(X_test_features)
        test_proba = self.model.predict_proba(X_test_features)[:, 1]
        
        # Store test results
        self.test_results = {
            'predictions': test_predictions,
            'probabilities': test_proba,
            'true_labels': y_test.values
        }
    
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
        
        # Create DataFrame with features and target, excluding player_name
        df = X.drop('player_name', axis=1).copy()
        df['is_injured'] = y
        
        # Get numerical and categorical features
        num_features = df.select_dtypes(include=['int64', 'float64']).columns
        cat_features = df.select_dtypes(include=['object', 'category']).columns
        
        # Remove target variable from features
        num_features = num_features.drop('is_injured') if 'is_injured' in num_features else num_features
        
        # Calculate statistics for each feature
        results = {}
        
        # Process numerical features
        for feature in num_features:
            try:
                # Skip if all values are the same
                if df[feature].nunique() <= 1:
                    logger.warning(f"Skipping feature {feature}: all values are the same")
                    continue
                    
                # Calculate median for binary split
                median = df[feature].median()
                high_risk = df[feature] > median
                
                # Create contingency table
                contingency = pd.crosstab(high_risk, df['is_injured'])
                
                # Skip if contingency table has any zeros
                if (contingency == 0).any().any():
                    logger.warning(f"Skipping feature {feature}: contingency table contains zeros")
                    continue
                
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
                    # Skip if category has too few samples
                    if (df[feature] == category).sum() < 10:  # Minimum 10 samples
                        continue
                        
                    # Create binary indicator
                    is_category = df[feature] == category
                    
                    # Create contingency table
                    contingency = pd.crosstab(is_category, df['is_injured'])
                    
                    # Skip if contingency table has any zeros
                    if (contingency == 0).any().any():
                        continue
                    
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
                    importance = feature_importances[list(self.feature_names).index(feature)]
                    
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
    for feature, stats in analysis_results.items():
        print(f"\n{feature}:")
        print(f"  Odds Ratio: {stats['odds_ratio']:.2f}")
        print(f"  Relative Risk: {stats['relative_risk']:.2f}")
        print(f"  Prevalence: {stats['prevalence']:.2%}")
        print(f"  Importance: {stats['importance']:.4f}")

if __name__ == "__main__":
    main() 