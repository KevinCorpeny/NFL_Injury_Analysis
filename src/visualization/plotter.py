from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
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
        plt.style.use('ggplot')
    
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
        
        # Create bar plot
        bars = plt.bar(range(len(injury_by_play)), injury_by_play.values)
        
        # Customize plot
        plt.title('Injury Rate by Play Type')
        plt.xlabel('Play Type')
        plt.ylabel('Injury Rate')
        plt.xticks(range(len(injury_by_play)), injury_by_play.index, rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2%}',
                    ha='center', va='bottom')
        
        self.save_plot('injury_by_play_type.png')
    
    def plot_injury_by_position(self, data: pd.DataFrame) -> None:
        """Plot injury frequency by player position.
        
        Args:
            data: DataFrame containing injury data
        """
        plt.figure(figsize=(12, 6))
        injury_by_position = data.groupby('position')['injury_occurred'].mean().sort_values(ascending=False)
        
        # Create bar plot
        bars = plt.bar(range(len(injury_by_position)), injury_by_position.values)
        
        # Customize plot
        plt.title('Injury Rate by Player Position')
        plt.xlabel('Position')
        plt.ylabel('Injury Rate')
        plt.xticks(range(len(injury_by_position)), injury_by_position.index, rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2%}',
                    ha='center', va='bottom')
        
        self.save_plot('injury_by_position.png')
    
    def plot_injury_severity_distribution(self, data: pd.DataFrame) -> None:
        """Plot distribution of injury severity.
        
        Args:
            data: DataFrame containing injury data
        """
        plt.figure(figsize=(10, 6))
        severity_counts = data['injury_severity'].value_counts()
        
        # Create bar plot
        bars = plt.bar(range(len(severity_counts)), severity_counts.values)
        
        # Customize plot
        plt.title('Distribution of Injury Severity')
        plt.xlabel('Severity')
        plt.ylabel('Count')
        plt.xticks(range(len(severity_counts)), severity_counts.index, rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    str(int(height)),
                    ha='center', va='bottom')
        
        self.save_plot('injury_severity_distribution.png')
    
    def plot_injury_trend_over_time(self, data: pd.DataFrame):
        """Plot injury trends over time."""
        try:
            # Create figure with adjusted size
            fig, ax = plt.subplots(figsize=(15, 8))
            
            # Create a proper time index for better x-axis
            data['time_index'] = data['season'] + (data['week'] - 1) / 17.0
            
            # Plot injury rate over time
            sns.lineplot(x='time_index', y='injury_rate', data=data, ax=ax, marker='o')
            
            # Customize the plot
            ax.set_title('NFL Injury Rate Over Time', pad=20, fontsize=16)
            ax.set_xlabel('Season', labelpad=10, fontsize=14)
            ax.set_ylabel('Injury Rate (per 1000 plays)', labelpad=10, fontsize=14)
            
            # Set x-axis ticks and labels
            seasons = sorted(data['season'].unique())
            tick_positions = []
            tick_labels = []
            
            # Add major ticks for each season
            for season in seasons:
                tick_positions.append(season)
                tick_labels.append(f'Season {int(season)}')
                
                # Add minor ticks for mid-season
                tick_positions.append(season + 0.5)
                tick_labels.append('Mid-Season')
            
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels, rotation=45, ha='right')
            
            # Add grid for better readability
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Adjust layout to prevent label cutoff
            plt.tight_layout()
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating injury trend plot: {str(e)}")
            raise
    
    def plot_injury_correlation_heatmap(self, data: pd.DataFrame) -> None:
        """Plot correlation heatmap of injury-related features.
        
        Args:
            data: DataFrame containing injury data
        """
        plt.figure(figsize=(12, 8))
        
        # Select numeric columns for correlation
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        correlation = data[numeric_cols].corr()
        
        # Create heatmap
        plt.imshow(correlation, cmap='coolwarm', interpolation='nearest')
        plt.colorbar()
        
        # Add labels
        plt.xticks(range(len(numeric_cols)), numeric_cols, rotation=45, ha='right')
        plt.yticks(range(len(numeric_cols)), numeric_cols)
        
        # Add correlation values
        for i in range(len(numeric_cols)):
            for j in range(len(numeric_cols)):
                plt.text(j, i, f'{correlation.iloc[i, j]:.2f}',
                        ha='center', va='center',
                        color='white' if abs(correlation.iloc[i, j]) > 0.5 else 'black')
        
        plt.title('Correlation Heatmap of Injury-Related Features')
        plt.tight_layout()
        
        self.save_plot('injury_correlation_heatmap.png')
    
    def plot_injury_by_game_situation(self, data: pd.DataFrame):
        """Plot injury rates by different game situations."""
        try:
            # Create subplots for different game situations with adjusted size
            fig, axes = plt.subplots(2, 2, figsize=(20, 15))
            fig.suptitle('NFL Injury Analysis by Game Situation', fontsize=16, y=0.95)
            
            # Plot 1: Injury rate by quarter
            quarter_data = data[['quarter', 'injury_rate_quarter']].drop_duplicates()
            quarter_data = quarter_data.sort_values('quarter')
            
            # Create bar plot for quarters
            sns.barplot(x='quarter', y='injury_rate_quarter', data=quarter_data, ax=axes[0, 0])
            axes[0, 0].set_title('Injury Rate by Quarter', pad=20)
            axes[0, 0].set_xlabel('Quarter', labelpad=10)
            axes[0, 0].set_ylabel('Injuries per 1000 Plays', labelpad=10)
            axes[0, 0].tick_params(axis='x', labelsize=14, pad=10)
            axes[0, 0].tick_params(axis='y', labelsize=12)
            
            # Set quarter labels
            quarter_labels = ['Q1', 'Q2', 'Q3', 'Q4', 'OT']
            num_quarters = len(quarter_data)
            axes[0, 0].set_xticks(range(num_quarters))
            axes[0, 0].set_xticklabels(quarter_labels[:num_quarters])
            
            # Add value labels on top of bars
            for i, rate in enumerate(quarter_data['injury_rate_quarter']):
                axes[0, 0].text(i, rate, f'{rate:.1f}', ha='center', va='bottom')
            
            # Add horizontal line for average
            avg_rate = quarter_data['injury_rate_quarter'].mean()
            axes[0, 0].axhline(y=avg_rate, color='red', linestyle='--', alpha=0.5)
            axes[0, 0].text(0, avg_rate, f'Avg: {avg_rate:.1f}', color='red', ha='left', va='bottom')
            
            # Plot 2: Injury rate by down
            down_data = data[['down', 'injury_rate_down']].drop_duplicates()
            down_data = down_data.sort_values('down')
            
            # Create bar plot for downs
            sns.barplot(x='down', y='injury_rate_down', data=down_data, ax=axes[0, 1])
            axes[0, 1].set_title('Injury Rate by Down', pad=20)
            axes[0, 1].set_xlabel('Down', labelpad=10)
            axes[0, 1].set_ylabel('Injuries per 1000 Plays', labelpad=10)
            axes[0, 1].tick_params(axis='x', labelsize=14, pad=10)
            axes[0, 1].tick_params(axis='y', labelsize=12)
            
            # Set down labels
            axes[0, 1].set_xticklabels(['1st', '2nd', '3rd', '4th'])
            
            # Add value labels on top of bars
            for i, rate in enumerate(down_data['injury_rate_down']):
                axes[0, 1].text(i, rate, f'{rate:.1f}', ha='center', va='bottom')
            
            # Add horizontal line for average
            avg_rate = down_data['injury_rate_down'].mean()
            axes[0, 1].axhline(y=avg_rate, color='red', linestyle='--', alpha=0.5)
            axes[0, 1].text(0, avg_rate, f'Avg: {avg_rate:.1f}', color='red', ha='left', va='bottom')
            
            # Plot 3: Injury rate by score differential
            score_data = data[['score_differential', 'injury_rate_score']].drop_duplicates()
            score_data = score_data.sort_values('score_differential')
            
            # Create scatter plot with trend line
            sns.scatterplot(x='score_differential', y='injury_rate_score', data=score_data, ax=axes[1, 0], s=100)
            sns.regplot(x='score_differential', y='injury_rate_score', data=score_data, ax=axes[1, 0], 
                       scatter=False, color='red', line_kws={'alpha': 0.5})
            
            axes[1, 0].set_title('Injury Rate by Score Differential', pad=20)
            axes[1, 0].set_xlabel('Score Differential (Points)', labelpad=10)
            axes[1, 0].set_ylabel('Injuries per 1000 Plays', labelpad=10)
            axes[1, 0].tick_params(axis='x', labelsize=12)
            axes[1, 0].tick_params(axis='y', labelsize=12)
            
            # Add reference line at 0
            axes[1, 0].axvline(x=0, color='gray', linestyle='--', alpha=0.5)
            
            # Add horizontal line for average
            avg_rate = score_data['injury_rate_score'].mean()
            axes[1, 0].axhline(y=avg_rate, color='red', linestyle='--', alpha=0.5)
            axes[1, 0].text(score_data['score_differential'].min(), avg_rate, 
                          f'Avg: {avg_rate:.1f}', color='red', ha='left', va='bottom')
            
            # Plot 4: Injury rate by time remaining
            time_data = data[['game_seconds_remaining', 'injury_rate_time']].drop_duplicates()
            time_data = time_data.sort_values('game_seconds_remaining')
            
            # Create scatter plot with trend line
            sns.scatterplot(x='game_seconds_remaining', y='injury_rate_time', data=time_data, ax=axes[1, 1], s=100)
            sns.regplot(x='game_seconds_remaining', y='injury_rate_time', data=time_data, ax=axes[1, 1],
                       scatter=False, color='red', line_kws={'alpha': 0.5})
            
            axes[1, 1].set_title('Injury Rate by Game Time Remaining', pad=20)
            axes[1, 1].set_xlabel('Game Time', labelpad=10)
            axes[1, 1].set_ylabel('Injuries per 1000 Plays', labelpad=10)
            axes[1, 1].tick_params(axis='x', labelsize=12)
            axes[1, 1].tick_params(axis='y', labelsize=12)
            
            # Format time remaining labels
            max_seconds = time_data['game_seconds_remaining'].max()
            tick_positions = [0, max_seconds/4, max_seconds/2, 3*max_seconds/4, max_seconds]
            tick_labels = ['End', 'Q4 Start', 'Half', 'Q2 Start', 'Start']
            axes[1, 1].set_xticks(tick_positions)
            axes[1, 1].set_xticklabels(tick_labels)
            
            # Add horizontal line for average
            avg_rate = time_data['injury_rate_time'].mean()
            axes[1, 1].axhline(y=avg_rate, color='red', linestyle='--', alpha=0.5)
            axes[1, 1].text(time_data['game_seconds_remaining'].min(), avg_rate,
                          f'Avg: {avg_rate:.1f}', color='red', ha='left', va='bottom')
            
            # Add grid to all subplots
            for ax in axes.flat:
                ax.grid(True, linestyle='--', alpha=0.7)
            
            # Adjust spacing between subplots
            plt.tight_layout(pad=3.0, h_pad=4.0, w_pad=4.0)
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating game situation plot: {str(e)}")
            raise
    
    def plot_injury_by_weather(self, data: pd.DataFrame) -> None:
        """Plot injury frequency by weather conditions.
        
        Args:
            data: DataFrame containing injury data
        """
        plt.figure(figsize=(12, 6))
        
        # Group by weather conditions
        weather_injury = data.groupby('weather')['injury_occurred'].mean().sort_values(ascending=False)
        
        # Create bar plot
        bars = plt.bar(range(len(weather_injury)), weather_injury.values)
        
        # Customize plot
        plt.title('Injury Rate by Weather Conditions')
        plt.xlabel('Weather')
        plt.ylabel('Injury Rate')
        plt.xticks(range(len(weather_injury)), weather_injury.index, rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2%}',
                    ha='center', va='bottom')
        
        self.save_plot('injury_by_weather.png') 