import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_analysis_results():
    """Load the analysis results from the model output."""
    model_path = Path("models/enhanced_risk_factor_model.joblib")
    params_path = Path("models/model_params.json")
    
    if not model_path.exists() or not params_path.exists():
        raise FileNotFoundError("Model files not found. Please run the risk analysis first.")
    
    with open(params_path, 'r') as f:
        params = json.load(f)
    
    return params

def create_feature_importance_plot(params):
    """Create a bar plot of feature importances."""
    plt.figure(figsize=(12, 8))
    
    # Extract feature importances
    features = []
    importances = []
    for feature, stats in params['analysis_results'].items():
        if feature.startswith(('body_region_', 'position_', 'game_type_', 'injury_severity_')):
            features.append(feature)
            importances.append(stats['importance'])
    
    # Sort by importance
    sorted_idx = np.argsort(importances)
    features = [features[i] for i in sorted_idx]
    importances = [importances[i] for i in sorted_idx]
    
    # Create the plot
    plt.barh(features, importances)
    plt.title('Feature Importances for Injury Risk Prediction')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig('visualizations/feature_importance.png')
    plt.close()

def create_body_region_heatmap(params):
    """Create a heatmap of odds ratios for body regions."""
    plt.figure(figsize=(10, 6))
    
    # Extract body region data
    body_regions = []
    odds_ratios = []
    prevalences = []
    
    for feature, stats in params['analysis_results'].items():
        if feature.startswith('body_region_'):
            body_regions.append(feature.replace('body_region_', ''))
            odds_ratios.append(stats['odds_ratio'])
            prevalences.append(stats['prevalence'])
    
    # Create DataFrame for heatmap
    data = pd.DataFrame({
        'Body Region': body_regions,
        'Odds Ratio': odds_ratios,
        'Prevalence': prevalences
    })
    
    # Create heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(data.set_index('Body Region')[['Odds Ratio', 'Prevalence']], 
                annot=True, fmt='.2f', cmap='YlOrRd')
    plt.title('Body Region Injury Risk Analysis')
    plt.tight_layout()
    plt.savefig('visualizations/body_region_heatmap.png')
    plt.close()

def create_position_risk_plot(params):
    """Create a scatter plot of position-specific risks."""
    plt.figure(figsize=(12, 8))
    
    # Extract position data
    positions = []
    odds_ratios = []
    prevalences = []
    
    for feature, stats in params['analysis_results'].items():
        if feature.startswith('position_'):
            positions.append(feature.replace('position_', ''))
            odds_ratios.append(stats['odds_ratio'])
            prevalences.append(stats['prevalence'])
    
    # Create scatter plot
    plt.scatter(prevalences, odds_ratios, s=100)
    for i, pos in enumerate(positions):
        plt.annotate(pos, (prevalences[i], odds_ratios[i]))
    
    plt.axhline(y=1, color='r', linestyle='--')
    plt.title('Position-Specific Injury Risk')
    plt.xlabel('Prevalence (%)')
    plt.ylabel('Odds Ratio')
    plt.tight_layout()
    plt.savefig('visualizations/position_risk.png')
    plt.close()

def create_severity_distribution(params):
    """Create a pie chart of injury severity distribution."""
    plt.figure(figsize=(10, 8))
    
    # Extract severity data
    severities = []
    prevalences = []
    
    for feature, stats in params['analysis_results'].items():
        if feature.startswith('injury_severity_'):
            severities.append(feature.replace('injury_severity_', ''))
            prevalences.append(stats['prevalence'])
    
    # Create pie chart
    plt.pie(prevalences, labels=severities, autopct='%1.1f%%')
    plt.title('Distribution of Injury Severity')
    plt.tight_layout()
    plt.savefig('visualizations/severity_distribution.png')
    plt.close()

def main():
    """Main function to create all visualizations."""
    # Create visualizations directory if it doesn't exist
    Path("visualizations").mkdir(exist_ok=True)
    
    try:
        # Load analysis results
        params = load_analysis_results()
        
        # Create visualizations
        logger.info("Creating feature importance plot...")
        create_feature_importance_plot(params)
        
        logger.info("Creating body region heatmap...")
        create_body_region_heatmap(params)
        
        logger.info("Creating position risk plot...")
        create_position_risk_plot(params)
        
        logger.info("Creating severity distribution plot...")
        create_severity_distribution(params)
        
        logger.info("All visualizations created successfully!")
        
    except Exception as e:
        logger.error(f"Error creating visualizations: {str(e)}")
        raise

if __name__ == "__main__":
    main() 