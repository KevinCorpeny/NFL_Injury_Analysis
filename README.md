# NFL Injury Analysis Project

This project analyzes NFL player injuries using comprehensive data analysis and machine learning techniques. The goal is to identify patterns and risk factors associated with player injuries in the NFL, with a focus on body regions, player positions, and game situations.

## Project Structure

```
nfl_injury_analysis/
├── config/                 # Configuration files
│   ├── validation_config.yaml  # Data validation rules
│   └── model_config.yaml      # Model configuration
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
│   │   ├── feature_engineering.py
│   │   └── validation.py      # Data validation system
│   ├── models/         # Model implementation
│   │   └── risk_factors.py  # ML model and risk analysis
│   └── utils/          # Utility functions
├── scripts/            # Scripts for running analysis
│   ├── run_risk_analysis.py  # ML model training and evaluation
│   └── visualize_risk_factors.py
├── tests/              # Test suite
│   ├── test_validation.py    # Validation system tests
│   └── test_model.py        # Model tests
├── examples/           # Example usage and demos
├── logs/              # Application logs
├── visualizations/    # Generated visualizations
├── requirements.txt    # Project dependencies
└── README.md          # Project documentation
```

## Features

### Data Validation System

The project implements a comprehensive data validation system that ensures data quality and consistency:

1. **Schema Validation**
   - Required columns presence
   - Unexpected columns detection
   - Data type verification

2. **Value Range Validation**
   - Season range (2000-2024)
   - Week range (1-18)
   - Game statistics ranges
   - Player participation metrics

3. **Consistency Rules**
   - Injury type and game status relationships
   - Player participation consistency
   - Injury severity and status alignment
   - Body region and injury type correlation

4. **Cross-field Validation**
   - Games played calculations
   - Injury timing metrics
   - Game situation consistency

5. **Statistical Validation**
   - Distribution checks
   - Anomaly detection
   - Statistical bounds verification

### Currently Implemented Analysis

* **B - Body Region Analysis**
  - Lower Body: Highest importance (0.2763), Odds Ratio 5.95, Prevalence 29.77%
  - Head/Neck: High Odds Ratio (30.44) but low Prevalence (3.80%)
  - Upper Body: Lower risk (Odds Ratio 0.95, Prevalence 7.63%)
  - Other: Moderate risk (Odds Ratio varies, Prevalence 5.46%)
  - Unknown: Lower risk (Odds Ratio 0.45, Prevalence 53.34%)

* **I - Injury Severity Analysis**
  - Mild to Moderate: 79.43% of injuries
  - Severe: 17.57% of injuries
  - Moderate to Severe: 3.00% of injuries
  - Comprehensive severity tracking
  - Recovery time analysis

* **N - Numerical Feature Engineering**
  - Injury week percentage calculation
  - Games played percentage
  - Time remaining bins
  - Score differential bins
  - Advanced statistical features

* **G - Game Context Analysis**
  - Quarter-specific risks
  - Down-specific patterns
  - Score differential impact
  - Time remaining analysis
  - Game type variations

* **O - Overall Model Performance**
  - Accuracy: 0.652
  - Precision: 0.844
  - Recall: 0.572
  - F1 Score: 0.682
  - AUC-ROC: 0.771

## Machine Learning Implementation

The project implements a robust machine learning pipeline for injury risk analysis:

1. **Data Preparation**
   - Feature selection and engineering
   - Handling missing values and outliers
   - Data splitting (80% training, 20% testing)
   - Comprehensive data validation

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

1. Run the data validation:
```bash
python scripts/validate_data.py
```

2. Run the risk analysis:
```bash
python scripts/run_risk_analysis.py
```

3. Generate visualizations:
```bash
python scripts/visualize_risk_factors.py
```

The visualizations will be saved in the `visualizations/` directory and include:
- Injury trend analysis over time
- Game situation analysis (quarter, down, score, time)
- Feature importance plots
- Body region heatmaps
- Position risk plots
- Severity distribution plots

## Technical Stack

* **Data Processing & Analysis**
  - Python 3.9+
  - pandas for data manipulation
  - numpy for numerical computations
  - scikit-learn for machine learning
  - imbalanced-learn for handling class imbalance

* **Data Validation**
  - Custom validation framework with rule-based validation
  - Schema validation (required columns, data types)
  - Value range validation (seasons, weeks, statistics)
  - Consistency rules (injury types, game status)
  - Cross-field validation (games played, timing metrics)

* **Data Visualization**
  - matplotlib for base plotting
  - seaborn for statistical visualizations
  - Static visualizations for:
    - Injury trends
    - Game situation analysis
    - Feature importance
    - Risk factor analysis
    - Severity distributions

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

3. **Validation Enhancement**
   - Add more sophisticated statistical validation
   - Implement real-time validation
   - Add validation rule learning
   - Improve validation performance

4. **Visualization Enhancement**
   - Basic statistical visualizations (matplotlib/seaborn)
   - Injury trend analysis over time
   - Game situation analysis (quarter, down, score, time)
   - Feature importance and risk factor visualization
   - Body region and position-specific risk analysis
   - Severity distribution visualization
   - Future plans:
     - Interactive dashboards
     - Real-time analysis capabilities
     - Advanced statistical visualizations
     - Model performance monitoring

5. **Analysis Extension**
   - Team-specific risk profiles
   - Season-long injury trends
   - Recovery time analysis
   - Predictive maintenance scheduling

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

* NFL data providers
* Contributors to the open-source libraries used in this project 