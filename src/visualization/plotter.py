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
    
    def plot_injury_trend_over_time(self, data: pd.DataFrame) -> plt.Figure:
        """Plot injury trends over time."""
        try:
            # Ensure time_index exists
            if 'time_index' not in data.columns:
                raise ValueError("Data must contain 'time_index' column")
            
            # Convert time index to numeric for plotting
            data['time_numeric'] = data['time_index'].apply(
                lambda x: float(x.split()[0]) + float(x.split()[1]) / 52  # Normalize weeks to fraction of year
            )
            
            # Sort by time
            data = data.sort_values('time_numeric')
            
            # Create figure with specific style
            plt.style.use('seaborn-v0_8')  # Use a valid style
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Set background color
            ax.set_facecolor('#f0f0f0')
            fig.patch.set_facecolor('white')
            
            # Plot injury rate over time with a thicker line and larger markers
            sns.lineplot(
                x='time_numeric', 
                y='injury_rate', 
                data=data, 
                ax=ax,
                marker='o',
                markersize=8,
                linewidth=2.5,
                color='#2077B4',  # A nice blue color
                label='Injury Rate'
            )
            
            # Add trend line
            z = np.polyfit(data['time_numeric'], data['injury_rate'], 1)
            p = np.poly1d(z)
            ax.plot(data['time_numeric'], p(data['time_numeric']), 
                   linestyle='--', color='#FF4B4B', alpha=0.7, linewidth=2,
                   label=f'Trend (slope: {z[0]:.1f} per year)')
            
            # Customize x-axis
            seasons = sorted(data['time_index'].str.split().str[0].unique())
            ax.set_xticks([float(s) for s in seasons])
            ax.set_xticklabels(seasons, rotation=45, ha='right')
            
            # Add labels and title with better formatting
            ax.set_xlabel('NFL Season', fontsize=12, fontweight='bold', labelpad=10)
            ax.set_ylabel('Injuries per 1000 Plays', fontsize=12, fontweight='bold', labelpad=10)
            ax.set_title('NFL Injury Rate Trend (2018-2022)', fontsize=14, fontweight='bold', pad=20)
            
            # Customize grid
            ax.grid(True, linestyle='--', alpha=0.3, color='gray')
            
            # Add legend with better formatting
            ax.legend(loc='upper left', frameon=True, framealpha=0.9, fontsize=10)
            
            # Adjust layout
            plt.tight_layout()
            
            # Set y-axis limits with some padding
            ymin = max(0, data['injury_rate'].min() * 0.9)
            ymax = data['injury_rate'].max() * 1.1
            ax.set_ylim(ymin, ymax)
            
            # Format y-axis ticks
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x)}'))
            
            # Add spines
            for spine in ax.spines.values():
                spine.set_color('#cccccc')
                spine.set_linewidth(1)
            
            # Customize tick parameters
            ax.tick_params(axis='both', which='major', labelsize=10)
            
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
    
    def plot_injury_by_game_situation(self, data: pd.DataFrame) -> plt.Figure:
        """Create a 2x2 grid of plots showing injury rates by game situation."""
        # Set style
        plt.style.use('seaborn-v0_8')
        
        # Create figure and subplots with more space between plots and at bottom
        fig, axes = plt.subplots(2, 2, figsize=(15, 16))  # Increased height to 16
        fig.suptitle('Injury Rates by Game Situation', fontsize=16, y=0.95)  # Moved title down
        
        # Plot quarter data
        quarter_data = data[data['situation'] == 'quarter'].copy()
        quarter_data['value'] = quarter_data['value'].astype(float)
        quarter_data = quarter_data.sort_values('value')
        
        sns.barplot(data=quarter_data, x='value', y='injury_rate', ax=axes[0, 0])
        axes[0, 0].set_title('Injury Rate by Quarter', pad=10)
        axes[0, 0].set_xlabel('Quarter')
        axes[0, 0].set_ylabel('Injuries per 1000 Plays')
        axes[0, 0].set_xticks(range(len(quarter_data)))
        axes[0, 0].set_xticklabels([f'Q{int(q)}' for q in quarter_data['value']])
        
        # Plot down data with dual axis
        down_data = data[data['situation'] == 'down'].copy()
        down_data['value'] = down_data['value'].astype(float)
        down_data = down_data.sort_values('value')
        
        # Create twin axis for play count
        ax_plays = axes[0, 1].twinx()
        
        # Plot injury rate bars
        sns.barplot(data=down_data, x='value', y='injury_rate', ax=axes[0, 1], color='#2077B4', alpha=0.7)
        # Plot play count line
        sns.lineplot(data=down_data, x=range(len(down_data)), y='play_count', ax=ax_plays, color='#FF4B4B', 
                    marker='o', linewidth=2, markersize=8)
        
        axes[0, 1].set_title('Injury Rate and Play Count by Down', pad=10)
        axes[0, 1].set_xlabel('Down')
        axes[0, 1].set_ylabel('Injuries per 1000 Plays')
        ax_plays.set_ylabel('Number of Plays', color='#FF4B4B')
        ax_plays.tick_params(axis='y', labelcolor='#FF4B4B')
        
        # Set ticks and labels for downs
        ordinal_suffixes = ['st', 'nd', 'rd', 'th']
        axes[0, 1].set_xticks(range(len(down_data)))
        axes[0, 1].set_xticklabels([
            f'{int(d)}{ordinal_suffixes[min(int(d)-1, 3)]}' 
            for d in down_data['value']
        ])
        
        # Plot score differential data
        score_data = data[data['situation'] == 'score_differential'].copy()
        sns.barplot(data=score_data, x='value', y='injury_rate', ax=axes[1, 0])
        axes[1, 0].set_title('Injury Rate by Score Differential', pad=20)
        axes[1, 0].set_xlabel('Point Differential')
        axes[1, 0].set_ylabel('Injuries per 1000 Plays')
        # Rotate and align the tick labels so they look better
        plt.setp(axes[1, 0].xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Plot time remaining data
        time_data = data[data['situation'] == 'time_remaining'].copy()
        sns.barplot(data=time_data, x='value', y='injury_rate', ax=axes[1, 1])
        axes[1, 1].set_title('Injury Rate by Time Remaining', pad=20)
        axes[1, 1].set_xlabel('Time Remaining (minutes)')
        axes[1, 1].set_ylabel('Injuries per 1000 Plays')
        
        # Add gridlines and set consistent y-axis limits for all subplots
        max_rate = data['injury_rate'].max()
        min_rate = data['injury_rate'].min()
        y_padding = (max_rate - min_rate) * 0.1
        
        # Format all subplots
        for ax in [axes[0, 0], axes[1, 0], axes[1, 1]]:
            ax.grid(True, alpha=0.3)
            ax.set_ylim(min_rate - y_padding, max_rate + y_padding)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}'))
            ax.margins(x=0.1)
        
        # Special handling for down plot with dual axis
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_ylim(min_rate - y_padding, max_rate + y_padding)
        axes[0, 1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}'))
        axes[0, 1].margins(x=0.1)
        
        # Format play count axis to use comma separator
        ax_plays.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
        
        # Add legend for down plot
        axes[0, 1].legend(['Injury Rate'], loc='upper left', frameon=True, framealpha=0.9)
        ax_plays.legend(['Play Count'], loc='upper right', frameon=True, framealpha=0.9)
        
        # First adjust the spacing between subplots
        plt.subplots_adjust(
            top=0.93,      # Move plots down from top
            bottom=0.18,   # Increase space at bottom
            hspace=0.45,   # Increase vertical space between plots
            wspace=0.25    # Space between columns
        )
        
        # Then adjust the overall layout
        plt.tight_layout(rect=[0, 0.05, 1, 0.93])
        
        # Move x-axis labels down for bottom plots
        for ax in [axes[1, 0], axes[1, 1]]:
            ax.tick_params(axis='x', pad=15)  # Add padding between axis and labels
        
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