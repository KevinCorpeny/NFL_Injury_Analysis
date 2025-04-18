# NFL Injury Analysis Project

This project analyzes NFL player injuries using comprehensive data analysis and machine learning techniques. The goal is to identify patterns and risk factors associated with player injuries in the NFL, with a focus on body regions, player positions, and game situations.

## Project Structure

```
nfl_injury_analysis/
├── config/                 # Configuration files
├── data/                  # Data storage
│   ├── raw/              # Raw data files
│   └── processed/        # Processed datasets
├── models/               # Model storage and versioning
│   ├── enhanced_risk_factor_model.joblib  # Trained ML model
│   └── model_params.json  # Model parameters and results
├── notebooks/            # Jupyter notebooks for analysis
├── reports/             # Generated reports and visualizations
├── src/                 # Source code
│   ├── data/           # Data processing modules
│   │   ├── preprocessing.py
│   │   └── feature_engineering.py
│   ├── models/         # Model implementation
│   │   └── risk_factors.py  # ML model and risk analysis
│   └── utils/          # Utility functions
├── scripts/            # Scripts for running analysis
│   ├── run_risk_analysis.py  # ML model training and evaluation
│   └── visualize_risk_factors.py
├── tests/              # Test suite
├── examples/           # Example usage and demos
├── logs/              # Application logs
├── visualizations/    # Generated visualizations
├── requirements.txt    # Project dependencies
└── README.md          # Project documentation
```

## Features

### Currently Implemented

* **Machine Learning Implementation**
  - Enhanced Risk Factor Analyzer with Random Forest classifier
  - Advanced preprocessing pipeline:
    - Numerical feature standardization
    - Categorical feature one-hot encoding
    - SMOTE for handling class imbalance
  - Comprehensive model evaluation:
    - Accuracy: 0.652
    - Precision: 0.844
    - Recall: 0.572
    - F1 Score: 0.682
    - AUC-ROC: 0.771
  - Feature importance analysis
  - Risk factor quantification
  - Model persistence and versioning

* **Data Processing Pipeline**
  - Robust data loading and preprocessing
  - Missing value handling (e.g., injury severity)
  - Outlier detection and treatment
  - Advanced feature engineering:
    - Injury week percentage calculation
    - Games played percentage
    - Time remaining bins
    - Score differential bins

* **Risk Factor Analysis**
  - Enhanced Risk Factor Analyzer implementation
  - Feature importance calculation
  - Odds ratio and relative risk analysis
  - Prevalence calculations
  - Comprehensive risk factor identification

* **Visualization System**
  - Feature importance plots
  - Body region heatmaps
  - Position risk scatter plots
  - Severity distribution charts
  - Automated visualization generation

* **Development Infrastructure**
  - Comprehensive logging system
  - Unit testing framework
  - Code coverage reporting
  - Type hints and static type checking
  - Modular project architecture

### Key Findings

1. **Body Region Analysis**
   - Lower Body: Highest importance (0.2763), Odds Ratio 5.95, Prevalence 29.77%
   - Head/Neck: High Odds Ratio (30.44) but low Prevalence (3.80%)
   - Upper Body: Lower risk (Odds Ratio 0.95, Prevalence 7.63%)

2. **Position-Specific Risks**
   - CB: Higher risk (Odds Ratio 1.22, Prevalence 12.26%)
   - QB: Lower risk (Odds Ratio 0.63, Prevalence 3.41%)
   - WR: Moderate risk (Odds Ratio 1.21, Prevalence 13.56%)

3. **Injury Severity Distribution**
   - Mild to Moderate: 79.43% of injuries
   - Severe: 17.57% of injuries
   - Moderate to Severe: 3.00% of injuries

## Machine Learning Implementation

The project implements a robust machine learning pipeline for injury risk analysis:

1. **Data Preparation**
   - Feature selection and engineering
   - Handling missing values and outliers
   - Data splitting (80% training, 20% testing)

2. **Model Architecture**
   - Random Forest classifier
   - SMOTE for class imbalance handling
   - StandardScaler for numerical features
   - OneHotEncoder for categorical features

3. **Model Performance**
   - High precision (0.844) indicates reliable positive predictions
   - Moderate recall (0.572) suggests room for improvement in capturing all injuries
   - Good AUC-ROC (0.771) shows strong discriminative ability
   - Balanced accuracy (0.652) considering class imbalance

4. **Feature Importance**
   - Body region features dominate importance scores
   - Game-related features show moderate importance
   - Position-specific features contribute to risk assessment

## Setup

1. Clone the repository:
```bash
git clone https://github.com/KevinCorpeny/NFL_Injury_Analysis.git
cd NFL_Injury_Analysis
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Linux/Mac
# or
.\venv\Scripts\activate  # On Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the risk analysis:
```bash
python scripts/run_risk_analysis.py
```

2. Generate visualizations:
```bash
python scripts/visualize_risk_factors.py
```

The visualizations will be saved in the `visualizations/` directory.

## Technical Stack

* **Data Processing & Analysis**
  - Python 3.9+
  - pandas for data manipulation
  - numpy for numerical computations
  - scikit-learn for machine learning
  - imbalanced-learn for handling class imbalance

* **Data Visualization**
  - matplotlib for base plotting
  - seaborn for statistical visualizations

* **Development Tools**
  - pytest for testing framework
  - coverage for code coverage
  - black for code formatting
  - mypy for static type checking

## Future Directions

1. **Model Enhancement**
   - Implement additional machine learning models
   - Develop ensemble methods
   - Add hyperparameter optimization
   - Improve recall through advanced sampling techniques

2. **Feature Engineering**
   - Include weather conditions
   - Add player tracking data
   - Incorporate team-specific factors
   - Add temporal features for injury patterns

3. **Visualization Enhancement**
   - Interactive dashboards
   - Real-time analysis capabilities
   - Advanced statistical visualizations
   - Model performance monitoring

4. **Analysis Extension**
   - Team-specific risk profiles
   - Season-long injury trends
   - Recovery time analysis
   - Predictive maintenance scheduling

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

* NFL data providers
* Contributors to the open-source libraries used in this project 