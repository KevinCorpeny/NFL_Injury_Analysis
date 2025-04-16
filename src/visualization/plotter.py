from pathlib import Path
from typing import Dict, List, Optional, Any
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from src.utils.logging import logger
import os
import matplotlib.gridspec as gridspec

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
    
    def save_plot(self, filename: str, dpi=None) -> None:
        """Save the current plot to a file.
        
        Args:
            filename (str): Name of the file to save the plot to
            dpi (int, optional): The resolution in dots per inch. Defaults to None.
        """
        plt.savefig(os.path.join(self.output_dir, filename), dpi=dpi)
        plt.close()
    
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
    
    def plot_injury_trend_over_time(self, data: pd.DataFrame) -> plt.Figure:
        """Plot injury trends over time."""
        try:
            # Ensure time_index exists
            if 'time_index' not in data.columns:
                raise ValueError("Data must contain 'time_index' column")
            
            # Create a copy of the data and sort by season and week
            data = data.copy()
            data = data.sort_values(['season', 'week'])
            
            # Calculate time values - each season takes up 20 units (18 weeks + 2 unit gap)
            min_season = min(data['season'])
            data['time_numeric'] = (data['season'] - min_season) * 20 + data['week']
            
            # Create figure with specific style
            plt.style.use('seaborn-v0_8')
            fig, ax1 = plt.subplots(figsize=(15, 8))
            
            # Set background color
            ax1.set_facecolor('#f0f0f0')
            fig.patch.set_facecolor('white')
            
            # Create second y-axis for total plays
            ax2 = ax1.twinx()
            
            # Plot injury rate and total plays
            line1 = ax1.plot(data['time_numeric'], data['injury_rate'], 
                           color='#FF4B4B', linewidth=2.5, marker='o', 
                           markersize=6, label='Injury Rate')
            
            line2 = ax2.plot(data['time_numeric'], data['total_plays'],
                           color='#2077B4', linewidth=2, linestyle='-',
                           alpha=0.7, label='Total Plays')
            
            # Add season markers and shading
            seasons = sorted(data['season'].unique())
            for season in seasons:
                # Calculate season boundaries
                season_start = (season - min_season) * 20
                season_end = season_start + 18  # Regular season weeks
                
                # Add shaded region for regular season
                ax1.axvspan(season_start, season_end, color='green', alpha=0.1)
                
                # Add vertical line for season start
                ax1.axvline(x=season_start, color='green', linestyle='--', alpha=0.5)
                ax1.text(season_start + 0.5, ax1.get_ylim()[1], f'{int(season)} Season',
                        rotation=90, verticalalignment='top', color='green', alpha=0.7)
                
                # Add "Off Season" text in the gap
                if season < max(seasons):
                    off_season_pos = season_end + 1
                    ax1.text(off_season_pos, ax1.get_ylim()[1] * 0.5, 'Off\nSeason',
                            ha='center', va='center', color='gray', alpha=0.7,
                            fontsize=10, fontweight='bold')
            
            # Customize x-axis with week ticks
            all_ticks = []
            all_labels = []
            
            for season in seasons:
                # Add tick for weeks 1, 5, 9, 13, 17
                base_week = (season - min_season) * 20
                for week in [1, 5, 9, 13, 17]:
                    tick = base_week + week
                    all_ticks.append(tick)
                    all_labels.append(f"Week {week}\n{int(season)}")
            
            ax1.set_xticks(all_ticks)
            ax1.set_xticklabels(all_labels, rotation=45, ha='right')
            
            # Add labels and title
            ax1.set_xlabel('Season and Week', fontsize=12, fontweight='bold', labelpad=10)
            ax1.set_ylabel('Injuries per 1000 Plays', fontsize=12, fontweight='bold', labelpad=10, color='#FF4B4B')
            ax2.set_ylabel('Total Plays', fontsize=12, fontweight='bold', labelpad=10, color='#2077B4')
            plt.title('NFL Injury Rate and Total Plays Over Time', fontsize=14, fontweight='bold', pad=20)
            
            # Customize grid
            ax1.grid(True, linestyle='--', alpha=0.3, color='gray')
            
            # Add combined legend
            lines = line1 + line2
            labels = ['Injury Rate', 'Total Plays']
            ax1.legend(lines, labels, loc='upper left', frameon=True, framealpha=0.9, fontsize=10)
            
            # Format y-axis ticks
            ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x)}'))
            ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
            ax1.tick_params(axis='y', labelcolor='#FF4B4B')
            ax2.tick_params(axis='y', labelcolor='#2077B4')
            
            # Set y-axis limits with padding
            ymin1 = max(0, data['injury_rate'].min() * 0.9)
            ymax1 = data['injury_rate'].max() * 1.1
            ax1.set_ylim(ymin1, ymax1)
            
            ymin2 = max(0, data['total_plays'].min() * 0.9)
            ymax2 = data['total_plays'].max() * 1.1
            ax2.set_ylim(ymin2, ymax2)
            
            # Add spines
            for spine in ax1.spines.values():
                spine.set_color('#cccccc')
                spine.set_linewidth(1)
            
            # Customize tick parameters
            ax1.tick_params(axis='both', which='major', labelsize=10)
            ax2.tick_params(axis='y', which='major', labelsize=10)
            
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
    
    def plot_injury_by_game_situation(self, quarter_data: pd.DataFrame, down_data: pd.DataFrame, 
                                    score_data: pd.DataFrame, time_data: pd.DataFrame) -> plt.Figure:
        """Create a 2x2 grid of plots showing injury rates by different game situations."""
        plt.style.use('seaborn-v0_8')
        fig = plt.figure(figsize=(15, 12))
        # Increase spacing between subplots
        plt.subplots_adjust(hspace=0.5, wspace=0.3)
        
        # Create a 2x2 grid of subplots
        ax1 = plt.subplot(221)
        ax2 = plt.subplot(222)
        ax3 = plt.subplot(223)
        ax4 = plt.subplot(224)

        # Quarter plot
        sns.barplot(data=quarter_data, x='quarter', y='injury_rate', ax=ax1, color='skyblue')
        x = quarter_data['quarter'].values
        y = quarter_data['injury_rate'].values
        if len(np.unique(y)) > 1:  # Only add trend line if there's variation
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            ax1.plot(x, p(x), "r--", alpha=0.8, label='Trend')
            ax1.legend()
        ax1.set_title('Injury Rate by Quarter', pad=15)
        ax1.set_xlabel('Quarter')
        ax1.set_ylabel('Injuries per 1000 Plays')
        ax1.tick_params(axis='x', rotation=0)

        # Down plot
        sns.barplot(data=down_data, x='down', y='injury_rate', ax=ax2, color='lightgreen')
        x = down_data['down'].values
        y = down_data['injury_rate'].values
        if len(np.unique(y)) > 1:  # Only add trend line if there's variation
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            ax2.plot(x, p(x), "r--", alpha=0.8, label='Trend')
            ax2.legend()
        ax2.set_title('Injury Rate by Down', pad=15)
        ax2.set_xlabel('Down')
        ax2.set_ylabel('Injuries per 1000 Plays')
        ax2.tick_params(axis='x', rotation=0)

        # Score differential plot
        score_data = score_data.sort_values('score_bin')
        sns.barplot(data=score_data, x='score_bin', y='injury_rate', ax=ax3, color='salmon')
        # Extract numeric values for trend line
        x = range(len(score_data))
        y = score_data['injury_rate'].values
        if len(np.unique(y)) > 1:  # Only add trend line if there's variation
            z = np.polyfit(x, y, 2)  # Quadratic fit for score differential
            p = np.poly1d(z)
            ax3.plot(x, p(x), "r--", alpha=0.8, label='Trend')
            ax3.legend()
        ax3.set_title('Injury Rate by Score Differential', pad=15)
        ax3.set_xlabel('Score Differential')
        ax3.set_ylabel('Injuries per 1000 Plays')
        # Improve x-axis readability for score differential
        ax3.tick_params(axis='x', rotation=30)
        score_labels = score_data['score_bin'].values
        ax3.set_xticks(range(len(score_labels)))
        # Show fewer labels for better readability
        visible_labels = ['-50 to -20', '-10 to -5', '0 to 5', '10 to 15', '20 to 50']
        ax3.set_xticklabels(score_labels)
        for idx, label in enumerate(ax3.get_xticklabels()):
            if label.get_text() not in visible_labels:
                label.set_visible(False)
        plt.setp(ax3.get_xticklabels(), ha='right')

        # Time remaining plot
        time_data = time_data.sort_values('time_bin', ascending=False)
        sns.barplot(data=time_data, x='time_bin', y='injury_rate', ax=ax4, color='lightpink')
        x = range(len(time_data))
        y = time_data['injury_rate'].values
        if len(np.unique(y)) > 1:  # Only add trend line if there's variation
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            ax4.plot(x, p(x), "r--", alpha=0.8, label='Trend')
            ax4.legend()
        ax4.set_title('Injury Rate by Time Remaining', pad=15)
        ax4.set_xlabel('Time Remaining (minutes)')
        ax4.set_ylabel('Injuries per 1000 Plays')
        # Improve x-axis readability for time remaining
        ax4.tick_params(axis='x', rotation=30)
        time_labels = time_data['time_bin'].values
        ax4.set_xticks(range(len(time_labels)))
        # Show only every third label
        for idx, label in enumerate(ax4.get_xticklabels()):
            if idx % 3 != 0:
                label.set_visible(False)
        plt.setp(ax4.get_xticklabels(), ha='right')

        plt.tight_layout()
        return fig
    
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