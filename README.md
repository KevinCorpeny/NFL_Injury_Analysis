# NFL Injury Analysis Project

This project analyzes the correlation between NFL play types and player injuries using nflfastR data and NFL injury reports. The goal is to identify patterns and risk factors associated with player injuries in the NFL.

## Project Structure

```
nfl_injury_analysis/
├── config/                 # Configuration files
│   └── default.yaml       # Default configuration settings
├── data/                  # Data storage
│   ├── injury_reports/    # NFL injury report data
│   ├── nfl_plays/        # Play-by-play data
│   └── processed/        # Processed and merged datasets
├── notebooks/            # Jupyter notebooks for analysis
├── reports/             # Generated reports and visualizations
├── src/                 # Source code
│   ├── data/           # Data processing modules
│   │   ├── base_processor.py
│   │   ├── injury_processor.py
│   │   └── nfl_play_processor.py
│   ├── utils/          # Utility functions
│   │   └── logging.py
│   ├── config.py       # Configuration management
│   └── process_data.py # Main processing script
├── scripts/            # Utility scripts
│   └── download_injury_data.py
├── requirements.txt    # Project dependencies
└── README.md          # Project documentation
```

## Features

- Data processing pipeline for NFL play-by-play and injury data
- Automated data validation and cleaning
- Feature engineering for injury analysis
- Configurable processing parameters
- Comprehensive logging system
- Modular and extensible architecture
- Advanced data visualizations:
  - Time trend analysis with playoff period indicators
  - Game situation analysis (quarter, down, score differential)
  - Dual-axis plots showing injury rates and play counts
  - Customizable plot styling and formatting

## Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/nfl_injury_analysis.git
cd nfl_injury_analysis
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

1. Download the required data:
```bash
python scripts/download_injury_data.py
```

2. Process the data:
```bash
python run_processing.py
```

3. Generate visualizations:
```bash
python run_visualizations.py
```
The visualizations will be saved in the `reports/figures/` directory.

4. Explore the data using Jupyter notebooks in the `notebooks/` directory.

## Data Sources

- **nflfastR**: Comprehensive play-by-play data for NFL games
  - Includes detailed play information, player statistics, and game context
  - Available at: https://www.nflfastr.com/

- **NFL Injury Reports**: Official injury data
  - Player injury status and details
  - Practice participation information
  - Game status designations

## Analysis Goals

1. **Play Type Analysis**
   - Identify play types with highest injury rates
   - Analyze risk factors in different play scenarios
   - Examine the impact of game situation on injury likelihood

2. **Position-Specific Analysis**
   - Analyze injury patterns by player position
   - Identify high-risk positions and situations
   - Study position-specific injury prevention strategies

3. **Temporal Analysis**
   - Examine injury patterns across seasons
   - Analyze weekly injury trends
   - Study the impact of scheduling on injury rates
   - Distinguish between regular season and playoff injury patterns
   - Track total play counts and their relationship with injury rates
   - Identify seasonal transitions and their effects on injury statistics

4. **Predictive Analysis**
   - Investigate correlation between play characteristics and injury likelihood
   - Develop risk assessment models
   - Identify preventive measures and best practices

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- NFL data provided by [NFLverse](https://nflverse.nflverse.com/)
- Special thanks to all contributors and maintainers of the open-source libraries used in this project 