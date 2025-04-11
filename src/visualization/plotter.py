from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils.logging import logger

class NFLPlotter:
    """Class for creating NFL injury analysis visualizations."""
    
    def __init__(self, output_dir: Path):
        """Initialize the plotter.
        
        Args:
            output_dir: Directory to save plots
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn')
        sns.set_palette("husl")
    
    def save_plot(self, filename: str) -> None:
        """Save the current plot.
        
        Args:
            filename: Name of the file to save
        """
        plt.tight_layout()
        plt.savefig(self.output_dir / filename)
        plt.close()
        logger.info(f"Saved plot to {filename}")
    
    def plot_injury_by_play_type(self, data: pd.DataFrame) -> None:
        """Plot injury frequency by play type.
        
        Args:
            data: DataFrame containing play and injury data
        """
        plt.figure(figsize=(12, 6))
        injury_by_play = data.groupby('play_type')['injury_occurred'].mean().sort_values(ascending=False)
        
        ax = sns.barplot(x=injury_by_play.index, y=injury_by_play.values)
        plt.title('Injury Rate by Play Type')
        plt.xlabel('Play Type')
        plt.ylabel('Injury Rate')
        plt.xticks(rotation=45)
        
        # Add value labels
        for i, v in enumerate(injury_by_play.values):
            ax.text(i, v, f'{v:.2%}', ha='center', va='bottom')
        
        self.save_plot('injury_by_play_type.png')
    
    def plot_injury_by_position(self, data: pd.DataFrame) -> None:
        """Plot injury frequency by player position.
        
        Args:
            data: DataFrame containing injury data
        """
        plt.figure(figsize=(12, 6))
        injury_by_position = data.groupby('position')['injury_occurred'].mean().sort_values(ascending=False)
        
        ax = sns.barplot(x=injury_by_position.index, y=injury_by_position.values)
        plt.title('Injury Rate by Player Position')
        plt.xlabel('Position')
        plt.ylabel('Injury Rate')
        plt.xticks(rotation=45)
        
        # Add value labels
        for i, v in enumerate(injury_by_position.values):
            ax.text(i, v, f'{v:.2%}', ha='center', va='bottom')
        
        self.save_plot('injury_by_position.png')
    
    def plot_injury_severity_distribution(self, data: pd.DataFrame) -> None:
        """Plot distribution of injury severity.
        
        Args:
            data: DataFrame containing injury data
        """
        plt.figure(figsize=(10, 6))
        severity_counts = data['injury_severity'].value_counts()
        
        ax = sns.barplot(x=severity_counts.index, y=severity_counts.values)
        plt.title('Distribution of Injury Severity')
        plt.xlabel('Severity')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        
        # Add value labels
        for i, v in enumerate(severity_counts.values):
            ax.text(i, v, str(v), ha='center', va='bottom')
        
        self.save_plot('injury_severity_distribution.png')
    
    def plot_injury_trend_over_time(self, data: pd.DataFrame) -> None:
        """Plot injury trend over time.
        
        Args:
            data: DataFrame containing injury data
        """
        plt.figure(figsize=(12, 6))
        
        # Group by week and calculate injury rate
        weekly_injuries = data.groupby('week')['injury_occurred'].mean()
        
        sns.lineplot(x=weekly_injuries.index, y=weekly_injuries.values)
        plt.title('Injury Rate Trend Over Season')
        plt.xlabel('Week')
        plt.ylabel('Injury Rate')
        plt.grid(True)
        
        self.save_plot('injury_trend_over_time.png')
    
    def plot_injury_correlation_heatmap(self, data: pd.DataFrame) -> None:
        """Plot correlation heatmap of injury-related features.
        
        Args:
            data: DataFrame containing injury and play data
        """
        plt.figure(figsize=(12, 10))
        
        # Select numeric columns for correlation
        numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
        correlation_matrix = data[numeric_cols].corr()
        
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Feature Correlation Heatmap')
        
        self.save_plot('injury_correlation_heatmap.png')
    
    def plot_injury_by_game_situation(self, data: pd.DataFrame) -> None:
        """Plot injury rates by game situation.
        
        Args:
            data: DataFrame containing play and injury data
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Quarter
        injury_by_quarter = data.groupby('quarter')['injury_occurred'].mean()
        sns.barplot(x=injury_by_quarter.index, y=injury_by_quarter.values, ax=axes[0, 0])
        axes[0, 0].set_title('Injury Rate by Quarter')
        
        # Down
        injury_by_down = data.groupby('down')['injury_occurred'].mean()
        sns.barplot(x=injury_by_down.index, y=injury_by_down.values, ax=axes[0, 1])
        axes[0, 1].set_title('Injury Rate by Down')
        
        # Score Differential
        injury_by_score_diff = data.groupby('score_differential_category')['injury_occurred'].mean()
        sns.barplot(x=injury_by_score_diff.index, y=injury_by_score_diff.values, ax=axes[1, 0])
        axes[1, 0].set_title('Injury Rate by Score Differential')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Time Remaining
        data['time_remaining_category'] = pd.cut(
            data['game_seconds_remaining'],
            bins=[0, 900, 1800, 2700, 3600],
            labels=['0-15min', '15-30min', '30-45min', '45-60min']
        )
        injury_by_time = data.groupby('time_remaining_category')['injury_occurred'].mean()
        sns.barplot(x=injury_by_time.index, y=injury_by_time.values, ax=axes[1, 1])
        axes[1, 1].set_title('Injury Rate by Time Remaining')
        
        plt.tight_layout()
        self.save_plot('injury_by_game_situation.png')
    
    def plot_injury_by_weather(self, data: pd.DataFrame) -> None:
        """Plot injury rates by weather conditions.
        
        Args:
            data: DataFrame containing play and injury data
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Temperature
        data['temp_category'] = pd.cut(
            data['temperature'],
            bins=[-float('inf'), 32, 50, 70, 90, float('inf')],
            labels=['<32°F', '32-50°F', '50-70°F', '70-90°F', '>90°F']
        )
        injury_by_temp = data.groupby('temp_category')['injury_occurred'].mean()
        sns.barplot(x=injury_by_temp.index, y=injury_by_temp.values, ax=axes[0])
        axes[0].set_title('Injury Rate by Temperature')
        
        # Surface
        injury_by_surface = data.groupby('surface')['injury_occurred'].mean()
        sns.barplot(x=injury_by_surface.index, y=injury_by_surface.values, ax=axes[1])
        axes[1].set_title('Injury Rate by Playing Surface')
        axes[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        self.save_plot('injury_by_weather.png') 