import pandas as pd
import nfl_data_py as nfl
import os
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def fetch_nfl_data(years=None):
    """
    Fetch NFL play-by-play data for specified years using nfl_data_py.
    
    Args:
        years (list): List of years to fetch data for. If None, fetches last 5 years.
    
    Returns:
        pd.DataFrame: Combined play-by-play data
    """
    if years is None:
        years = list(range(2018, 2024))  # Last 5 years
    
    data_dir = Path(__file__).parent.parent / 'data'
    data_dir.mkdir(exist_ok=True)
    
    all_data = []
    for year in years:
        logging.info(f"Fetching data for {year}...")
        try:
            # Fetch play-by-play data
            pbp_data = nfl.import_pbp_data([year])
            
            # Save raw data
            pbp_data.to_csv(data_dir / f'pbp_{year}.csv', index=False)
            
            all_data.append(pbp_data)
            logging.info(f"Successfully fetched data for {year}")
        except Exception as e:
            logging.error(f"Error fetching data for {year}: {str(e)}")
    
    if all_data:
        combined_data = pd.concat(all_data, ignore_index=True)
        combined_data.to_csv(data_dir / 'combined_pbp_data.csv', index=False)
        return combined_data
    else:
        logging.error("No data was successfully fetched")
        return None

if __name__ == "__main__":
    # Fetch data for the last 5 years
    data = fetch_nfl_data()
    if data is not None:
        logging.info(f"Successfully fetched {len(data)} plays") 