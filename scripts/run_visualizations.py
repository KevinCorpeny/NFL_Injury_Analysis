#!/usr/bin/env python3
"""
NFL Injury Analysis - Visualization Script

This script generates visualizations for NFL injury analysis using matplotlib.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from typing import Dict, List, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NFLPlotter:
    """Class for creating NFL injury analysis visualizations."""
    
    def __init__(self, data_dir: str = "data/processed"):
        """Initialize the plotter with data directory."""
        self.data_dir = Path(data_dir)
        self.plays_df = None
        self.injury_df = None
        self._load_data()
        
        # Set matplotlib style
        plt.style.use('ggplot')
        
    def _load_data(self) -> None:
        """Load the processed data files."""
        # Load play data
        play_data_path = self.data_dir / "processed_plays.parquet"
        if play_data_path.exists():
            self.plays_df = pd.read_parquet(play_data_path)
            logger.info(f"Loaded {len(self.plays_df)} plays")
        else:
            logger.error(f"Play data not found at {play_data_path}")
            
        # Load injury data
        injury_data_path = self.data_dir / "processed_injuries.parquet"
        if injury_data_path.exists():
            self.injury_df = pd.read_parquet(injury_data_path)
            logger.info(f"Loaded {len(self.injury_df)} injuries")
        else:
            logger.error(f"Injury data not found at {injury_data_path}")
    
    def plot_play_type_distribution(self, save_path: str = None) -> None:
        """Plot the distribution of play types."""
        if self.plays_df is None:
            logger.error("No play data available")
            return
            
        # Count play types
        play_counts = self.plays_df['play_type'].value_counts()
        
        # Create figure
        plt.figure(figsize=(12, 6))
        
        # Create bar plot
        bars = plt.bar(play_counts.index, play_counts.values)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height):,}',
                    ha='center', va='bottom')
        
        # Customize plot
        plt.title('Distribution of Play Types')
        plt.xlabel('Play Type')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save or show
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Saved play type distribution plot to {save_path}")
        else:
            plt.show()
        plt.close()
    
    def plot_yards_gained_distribution(self, save_path: str = None) -> None:
        """Plot the distribution of yards gained."""
        if self.plays_df is None:
            logger.error("No play data available")
            return
            
        # Create figure
        plt.figure(figsize=(12, 6))
        
        # Create histogram
        n, bins, patches = plt.hist(self.plays_df['yards_gained'], bins=70, 
                                  weights=np.ones(len(self.plays_df)) / len(self.plays_df) * 100)
        
        # Add percentage labels
        for i in range(len(patches)):
            if n[i] > 0.5:  # Only label significant bins
                plt.text(bins[i] + (bins[i+1] - bins[i])/2, n[i],
                        f'{n[i]:.1f}%',
                        ha='center', va='bottom')
        
        # Customize plot
        plt.title('Distribution of Yards Gained per Play')
        plt.xlabel('Yards Gained')
        plt.ylabel('% Of Plays')
        plt.xlim(-10, 50)
        plt.axvline(x=0, color='red', linestyle='--', alpha=0.5)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save or show
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Saved yards gained distribution plot to {save_path}")
        else:
            plt.show()
        plt.close()
    
    def plot_injury_timing(self, save_path: str = None) -> None:
        """Plot injury distribution by games played."""
        if self.injury_df is None:
            logger.error("No injury data available")
            return
            
        # Create figure
        plt.figure(figsize=(12, 6))
        
        # Create histogram
        n, bins, patches = plt.hist(self.injury_df['games_played_before_injury'], 
                                  bins=17,  # 17 weeks in NFL season
                                  weights=np.ones(len(self.injury_df)) / len(self.injury_df) * 100)
        
        # Add percentage labels
        for i in range(len(patches)):
            if n[i] > 0.5:  # Only label significant bins
                plt.text(bins[i] + (bins[i+1] - bins[i])/2, n[i],
                        f'{n[i]:.1f}%',
                        ha='center', va='bottom')
        
        # Customize plot
        plt.title('Distribution of Injuries by Games Played')
        plt.xlabel('Games Played Before Injury')
        plt.ylabel('% Of Injuries')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save or show
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Saved injury timing plot to {save_path}")
        else:
            plt.show()
        plt.close()
    
    def plot_injury_severity_by_games(self, save_path: str = None) -> None:
        """Plot injury severity by games played percentage."""
        if self.injury_df is None:
            logger.error("No injury data available")
            return
            
        # Create figure
        plt.figure(figsize=(12, 6))
        
        # Group data by severity
        severity_groups = self.injury_df.groupby('injury_severity')
        
        # Create box plot manually
        positions = range(len(severity_groups))
        for i, (severity, group) in enumerate(severity_groups):
            # Calculate box plot statistics
            q1 = group['games_played_percentage'].quantile(0.25)
            q3 = group['games_played_percentage'].quantile(0.75)
            iqr = q3 - q1
            lower_whisker = max(q1 - 1.5 * iqr, group['games_played_percentage'].min())
            upper_whisker = min(q3 + 1.5 * iqr, group['games_played_percentage'].max())
            
            # Draw box
            plt.boxplot(group['games_played_percentage'], 
                       positions=[i],
                       widths=0.6,
                       patch_artist=True,
                       boxprops=dict(facecolor=f'C{i}'),
                       medianprops=dict(color='black'))
        
        # Customize plot
        plt.title('Injury Severity by Games Played Percentage')
        plt.xlabel('Injury Severity')
        plt.ylabel('Percentage of Games Played')
        plt.xticks(positions, severity_groups.groups.keys(), rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save or show
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Saved injury severity plot to {save_path}")
        else:
            plt.show()
        plt.close()

def main():
    """Main function to run all visualizations."""
    # Create output directory
    output_dir = Path("visualizations")
    output_dir.mkdir(exist_ok=True)
    
    # Initialize plotter
    plotter = NFLPlotter()
    
    # Generate all visualizations
    plotter.plot_play_type_distribution(output_dir / "play_type_distribution.png")
    plotter.plot_yards_gained_distribution(output_dir / "yards_gained_distribution.png")
    plotter.plot_injury_timing(output_dir / "injury_timing.png")
    plotter.plot_injury_severity_by_games(output_dir / "injury_severity_by_games.png")

if __name__ == "__main__":
    main() 