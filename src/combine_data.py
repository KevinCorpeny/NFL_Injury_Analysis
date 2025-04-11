import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_play_by_play_data(years=None):
    """
    Load play-by-play data from CSV files.
    
    Args:
        years (list): List of years to load. If None, uses last 5 years.
    
    Returns:
        pd.DataFrame: Combined play-by-play data
    """
    if years is None:
        years = list(range(2018, 2024))
    
    data_dir = Path(__file__).parent.parent / 'data'
    all_data = []
    
    for year in years:
        file_path = data_dir / f'pbp_{year}.csv'
        if file_path.exists():
            try:
                df = pd.read_csv(file_path)
                df['season'] = year
                all_data.append(df)
                logging.info(f"Successfully loaded play-by-play data for {year}")
            except Exception as e:
                logging.error(f"Error loading play-by-play data for {year}: {str(e)}")
    
    if not all_data:
        logging.error("No play-by-play data was loaded")
        return None
    
    return pd.concat(all_data, ignore_index=True)

def load_injury_data(years=None):
    """
    Load injury data from CSV files.
    
    Args:
        years (list): List of years to load. If None, uses last 5 years.
    
    Returns:
        pd.DataFrame: Combined injury data
    """
    if years is None:
        years = list(range(2018, 2024))
    
    data_dir = Path(__file__).parent.parent / 'data'
    all_data = []
    
    for year in years:
        file_path = data_dir / f'injuries_{year}.csv'
        if file_path.exists():
            try:
                df = pd.read_csv(file_path)
                df['season'] = year
                all_data.append(df)
                logging.info(f"Successfully loaded injury data for {year}")
            except Exception as e:
                logging.error(f"Error loading injury data for {year}: {str(e)}")
    
    if not all_data:
        logging.error("No injury data was loaded")
        return None
    
    return pd.concat(all_data, ignore_index=True)

def combine_datasets(pbp_data, injury_data):
    """
    Combine play-by-play and injury data into a single dataset.
    
    Args:
        pbp_data (pd.DataFrame): Play-by-play data
        injury_data (pd.DataFrame): Injury data
    
    Returns:
        pd.DataFrame: Combined dataset
    """
    if pbp_data is None or injury_data is None:
        logging.error("Missing required data for combination")
        return None
    
    try:
        # Convert date columns to datetime
        pbp_data['game_date'] = pd.to_datetime(pbp_data['game_date'])
        injury_data['date'] = pd.to_datetime(injury_data['date'])
        
        # Create a mapping between player names and their positions
        player_positions = injury_data[['player_name', 'position']].drop_duplicates()
        
        # Merge play-by-play data with player positions
        combined_data = pd.merge(
            pbp_data,
            player_positions,
            left_on='player_name',
            right_on='player_name',
            how='left'
        )
        
        # Add injury information
        injury_plays = pd.merge(
            combined_data,
            injury_data[['player_name', 'injury_type', 'date', 'season']],
            left_on=['player_name', 'game_date', 'season'],
            right_on=['player_name', 'date', 'season'],
            how='left'
        )
        
        # Clean up the combined dataset
        injury_plays = injury_plays.drop(columns=['date'])
        injury_plays['injury_flag'] = injury_plays['injury_type'].notna().astype(int)
        
        # Add additional derived features
        injury_plays['days_since_last_injury'] = injury_plays.groupby('player_name')['game_date'].diff().dt.days
        injury_plays['injury_count'] = injury_plays.groupby('player_name')['injury_flag'].cumsum()
        
        logging.info("Successfully combined datasets")
        return injury_plays
    
    except Exception as e:
        logging.error(f"Error combining datasets: {str(e)}")
        return None

def save_combined_data(df, filename='combined_dataset.csv'):
    """
    Save the combined dataset to a CSV file.
    
    Args:
        df (pd.DataFrame): Combined dataset
        filename (str): Output filename
    """
    if df is None:
        logging.error("No data to save")
        return
    
    try:
        data_dir = Path(__file__).parent.parent / 'data'
        output_path = data_dir / filename
        df.to_csv(output_path, index=False)
        logging.info(f"Successfully saved combined dataset to {output_path}")
    except Exception as e:
        logging.error(f"Error saving combined dataset: {str(e)}")

def main():
    """Main function to combine play-by-play and injury data."""
    logging.info("Starting data combination process")
    
    # Load data
    pbp_data = load_play_by_play_data()
    injury_data = load_injury_data()
    
    # Combine datasets
    combined_data = combine_datasets(pbp_data, injury_data)
    
    if combined_data is not None:
        # Save the combined dataset
        save_combined_data(combined_data)
        
        # Print summary statistics
        logging.info("\nDataset Summary:")
        logging.info(f"Total plays: {len(combined_data)}")
        logging.info(f"Total injuries: {combined_data['injury_flag'].sum()}")
        logging.info(f"Injury rate: {combined_data['injury_flag'].mean():.4%}")
        
        # Print sample of injury data
        logging.info("\nSample of injury data:")
        injury_sample = combined_data[combined_data['injury_flag'] == 1].head()
        print(injury_sample[['player_name', 'position', 'injury_type', 'game_date']])
    
    logging.info("Data combination process completed")

if __name__ == "__main__":
    main() 