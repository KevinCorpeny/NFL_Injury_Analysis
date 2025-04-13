import sys
from pathlib import Path
import pandas as pd
import nfl_data_py as nfl
import matplotlib.pyplot as plt
import seaborn as sns

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.utils.logging import logger

def download_injury_data():
    """Download and prepare NFL injury data."""
    try:
        # Create directories
        data_dir = Path("data/injury_reports")
        data_dir.mkdir(parents=True, exist_ok=True)
        
        # Define seasons to download
        seasons = [2018, 2019, 2020, 2021, 2022]
        
        logger.info(f"Downloading NFL injury data for seasons {seasons}")
        
        # Download injury data
        injury_data = nfl.import_injuries(seasons)
        
        if injury_data is not None and not injury_data.empty:
            logger.info(f"Successfully downloaded {len(injury_data)} injury records")
            
            # Print available columns
            logger.info("Available columns in raw injury data:")
            logger.info(injury_data.columns.tolist())
            
            # Process and clean the data
            injury_data = process_injury_data(injury_data)
            
            # Save the data
            output_path = data_dir / "nfl_injuries.csv"
            injury_data.to_csv(output_path, index=False)
            
            logger.info(f"Injury data saved to {output_path}")
            
            # Print some statistics
            logger.info("\nInjury Data Statistics:")
            logger.info(f"Total injuries: {len(injury_data)}")
            
            # Print statistics based on available columns
            for col in ['game_type', 'report_status', 'practice_status', 'position']:
                if col in injury_data.columns:
                    logger.info(f"\nInjuries by {col}:")
                    logger.info(injury_data[col].value_counts().head())
            
            if 'plays_df' not in locals():
                print("Error: plays_df DataFrame not found. Please load the data first.")
            else:
                # Check for required columns
                required_columns = ['down', 'yards_to_go']
                missing_columns = [col for col in required_columns if col not in plays_df.columns]
                
                if missing_columns:
                    print(f"Missing required columns: {missing_columns}")
                    print("Available columns:", plays_df.columns.tolist())
                else:
                    # All required columns exist, proceed with visualization
                    plt.figure(figsize=(12, 6))
                    sns.boxplot(x='down', y='yards_to_go', data=plays_df)
                    plt.title('Distribution of Yards to Go by Down')
                    plt.xlabel('Down')
                    plt.ylabel('Yards to Go')
                    plt.tight_layout()
                    plt.show()
            
        else:
            logger.error("No injury data was downloaded")
            
    except Exception as e:
        logger.error(f"Error downloading injury data: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise

def process_injury_data(df: pd.DataFrame) -> pd.DataFrame:
    """Process and clean the injury data.
    
    Args:
        df: Raw injury data DataFrame
        
    Returns:
        Processed DataFrame
    """
    # Standardize column names
    df.columns = df.columns.str.lower()
    
    # Convert date columns
    date_columns = ['report_date']
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
    
    # Standardize positions
    if 'position' in df.columns:
        df['position'] = df['position'].str.upper()
    
    # Map game status to severity if available
    if 'game_status' in df.columns:
        status_map = {
            'Out': 'Severe',
            'Doubtful': 'Moderate',
            'Questionable': 'Minor',
            'Not Listed': 'None'
        }
        df['injury_severity'] = df['game_status'].map(status_map)
    
    # Keep all available columns
    return df

if __name__ == "__main__":
    download_injury_data() 