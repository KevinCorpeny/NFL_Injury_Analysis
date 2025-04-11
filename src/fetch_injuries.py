import pandas as pd
import requests
from pathlib import Path
from datetime import datetime, timedelta
import time
import json

class NFLInjuryFetcher:
    def __init__(self):
        self.base_url = "https://www.nfl.com/injuries/league/"
        self.data_dir = Path(__file__).parent.parent / 'data'
        self.data_dir.mkdir(exist_ok=True)

    def fetch_weekly_injuries(self, season, week):
        """
        Fetch injury reports for a specific week and season.
        
        Args:
            season (int): NFL season year
            week (int): Week number
            
        Returns:
            dict: Injury report data
        """
        try:
            # NFL's injury reports are typically released on Wednesday, Thursday, and Friday
            # We'll fetch the Friday report as it's most complete
            url = f"{self.base_url}/week/{week}/{season}"
            response = requests.get(url)
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Error fetching data for Week {week}, {season}: {response.status_code}")
                return None
        except Exception as e:
            print(f"Exception while fetching injury data: {str(e)}")
            return None

    def process_injury_data(self, raw_data):
        """
        Process raw injury report data into a structured format.
        
        Args:
            raw_data (dict): Raw injury report data
            
        Returns:
            pd.DataFrame: Processed injury data
        """
        if not raw_data:
            return pd.DataFrame()

        injuries = []
        for team_data in raw_data.get('injuries', []):
            team = team_data.get('team', '')
            for player in team_data.get('players', []):
                injury_info = {
                    'team': team,
                    'player_name': player.get('name', ''),
                    'position': player.get('position', ''),
                    'injury_type': player.get('injury', ''),
                    'practice_status': player.get('practice_status', ''),
                    'game_status': player.get('game_status', '')
                }
                injuries.append(injury_info)
        
        return pd.DataFrame(injuries)

    def fetch_season_injuries(self, season, start_week=1, end_week=18):
        """
        Fetch injury reports for an entire season.
        
        Args:
            season (int): NFL season year
            start_week (int): Starting week number
            end_week (int): Ending week number
            
        Returns:
            pd.DataFrame: Combined injury data for the season
        """
        all_injuries = []
        
        for week in range(start_week, end_week + 1):
            print(f"Fetching injuries for Week {week}, {season}")
            raw_data = self.fetch_weekly_injuries(season, week)
            
            if raw_data:
                df = self.process_injury_data(raw_data)
                df['season'] = season
                df['week'] = week
                all_injuries.append(df)
            
            # Be nice to the API
            time.sleep(1)
        
        if all_injuries:
            combined_df = pd.concat(all_injuries, ignore_index=True)
            output_file = self.data_dir / f'injuries_{season}.csv'
            combined_df.to_csv(output_file, index=False)
            return combined_df
        
        return pd.DataFrame()

if __name__ == "__main__":
    fetcher = NFLInjuryFetcher()
    # Fetch last 5 seasons of injury data
    current_year = datetime.now().year
    for year in range(current_year - 5, current_year + 1):
        print(f"\nFetching injury data for {year} season...")
        injury_data = fetcher.fetch_season_injuries(year)
        if not injury_data.empty:
            print(f"Successfully fetched {len(injury_data)} injury records for {year}") 