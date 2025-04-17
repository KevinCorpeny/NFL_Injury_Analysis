#!/usr/bin/env python3
"""
Script to run risk factor analysis for NFL injuries.
"""

import sys
from pathlib import Path
import importlib

# Add src directory to Python path
src_path = Path(__file__).parent.parent / 'src'
sys.path.append(str(src_path))

# Import and reload the module to ensure we're using the latest code
from models import risk_factors
importlib.reload(risk_factors)
from models.risk_factors import RiskFactorAnalyzer

import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Changed to DEBUG for more detailed output
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Run the risk factor analysis."""
    try:
        # Initialize analyzer
        logger.debug("Initializing RiskFactorAnalyzer...")
        analyzer = RiskFactorAnalyzer()
        logger.debug(f"Data directory: {analyzer.data_dir}")
        
        # Load and prepare data
        logger.info("Loading and preparing data...")
        df = analyzer.load_data()
        X, y = analyzer.prepare_features(df)
        logger.debug(f"Data loaded successfully. Shape: {df.shape}")
        logger.debug(f"Features prepared. X shape: {X.shape}, y shape: {y.shape}")
        
        # Train model
        logger.info("Training model...")
        analyzer.train_model(X, y)
        
        # Identify risk factors
        logger.info("Identifying risk factors...")
        risk_factors = analyzer.identify_risk_factors()
        
        # Analyze risk factors
        logger.info("Analyzing risk factors...")
        analysis_results = analyzer.analyze_risk_factors(X, y)
        
        # Save results
        logger.info("Saving results...")
        analyzer.save_model()
        
        # Print results
        print("\nIdentified Risk Factors:")
        for factor in risk_factors:
            print(f"\n{factor}:")
            print(f"  Odds Ratio: {analysis_results[factor]['odds_ratio']:.2f}")
            print(f"  Relative Risk: {analysis_results[factor]['relative_risk']:.2f}")
            print(f"  Prevalence: {analysis_results[factor]['prevalence']:.2%}")
        
    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}", exc_info=True)  # Added exc_info for stack trace
        sys.exit(1)

if __name__ == "__main__":
    main() 