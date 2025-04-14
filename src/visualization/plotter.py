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
            plt.style.use('seaborn-v0_8')
            fig, ax1 = plt.subplots(figsize=(12, 8))
            
            # Set background color
            ax1.set_facecolor('#f0f0f0')
            fig.patch.set_facecolor('white')
            
            # Create second y-axis for total plays
            ax2 = ax1.twinx()
            
            # Plot injury rate (blue line)
            line1 = ax1.plot(data['time_numeric'], data['injury_rate'], 
                           color='#2077B4', linewidth=2.5, marker='o', 
                           markersize=6, label='Injury Rate')
            
            # Plot total plays (red line)
            line2 = ax2.plot(data['time_numeric'], data['total_plays'],
                           color='#FF4B4B', linewidth=2, linestyle='-',
                           alpha=0.7, label='Total Plays')
            
            # Add vertical lines for playoff starts (Week 18 in each season)
            seasons = sorted(data['time_index'].str.split().str[0].unique())
            for season in seasons:
                playoff_start = float(season) + 18/52  # Week 18 is when regular season ends
                ax1.axvline(x=playoff_start, color='green', linestyle='--', alpha=0.5)
                # Add text annotation
                ax1.text(playoff_start + 0.02, ax1.get_ylim()[1], 'Playoffs Start',
                        rotation=90, verticalalignment='top', color='green', alpha=0.7)
            
            # Customize x-axis
            ax1.set_xticks([float(s) for s in seasons])
            ax1.set_xticklabels(seasons, rotation=45, ha='right')
            
            # Add labels and title
            ax1.set_xlabel('NFL Season', fontsize=12, fontweight='bold', labelpad=10)
            ax1.set_ylabel('Injuries per 1000 Plays', fontsize=12, fontweight='bold', labelpad=10)
            ax2.set_ylabel('Total Plays', fontsize=12, fontweight='bold', labelpad=10, color='#FF4B4B')
            plt.title('NFL Injury Rate and Total Plays Over Time', fontsize=14, fontweight='bold', pad=20)
            
            # Customize grid
            ax1.grid(True, linestyle='--', alpha=0.3, color='gray')
            
            # Add combined legend
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax1.legend(lines, labels, loc='upper left', frameon=True, framealpha=0.9, fontsize=10)
            
            # Format y-axis ticks
            ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x)}'))
            ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
            ax2.tick_params(axis='y', labelcolor='#FF4B4B')
            
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
    
    def plot_injury_by_game_situation(self, data: pd.DataFrame) -> plt.Figure:
        """Create a 2x2 grid of plots showing injury rates by game situation."""
        # Set style
        plt.style.use('seaborn-v0_8')
        
        # Create figure and subplots with more space between plots and at bottom
        fig = plt.figure(figsize=(15, 16))  # Increased height to 16
        gs = plt.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.3)  # Increased spacing
        axes = [
            fig.add_subplot(gs[0, 0]),
            fig.add_subplot(gs[0, 1]),
            fig.add_subplot(gs[1, 0]),
            fig.add_subplot(gs[1, 1])
        ]
        
        fig.suptitle('Injury Rates by Game Situation', fontsize=16, y=0.95)  # Moved title up
        
        # Plot quarter data
        quarter_data = data[data['situation'] == 'quarter'].copy()
        quarter_data['value'] = quarter_data['value'].astype(float)
        quarter_data = quarter_data.sort_values('value')
        
        sns.barplot(data=quarter_data, x='value', y='injury_rate', ax=axes[0])
        axes[0].set_title('Injury Rate by Quarter', pad=15)
        axes[0].set_xlabel('Quarter', labelpad=10)
        axes[0].set_ylabel('Injuries per 1000 Plays', labelpad=10)
        axes[0].set_xticks(range(len(quarter_data)))
        axes[0].set_xticklabels([f'Q{int(q)}' for q in quarter_data['value']])
        
        # Plot down data with dual axis
        down_data = data[data['situation'] == 'down'].copy()
        down_data['value'] = down_data['value'].astype(float)
        down_data = down_data.sort_values('value')
        
        # Create twin axis for play count
        ax_plays = axes[1].twinx()
        
        # Plot injury rate bars
        sns.barplot(data=down_data, x='value', y='injury_rate', ax=axes[1], color='#2077B4', alpha=0.7)
        # Plot play count line
        sns.lineplot(data=down_data, x=range(len(down_data)), y='play_count', ax=ax_plays, color='#FF4B4B', 
                    marker='o', linewidth=2, markersize=8)
        
        axes[1].set_title('Injury Rate and Play Count by Down', pad=15)
        axes[1].set_xlabel('Down', labelpad=10)
        axes[1].set_ylabel('Injuries per 1000 Plays', labelpad=10)
        ax_plays.set_ylabel('Number of Plays', color='#FF4B4B', labelpad=10)
        ax_plays.tick_params(axis='y', labelcolor='#FF4B4B')
        
        # Set ticks and labels for downs
        ordinal_suffixes = ['st', 'nd', 'rd', 'th']
        axes[1].set_xticks(range(len(down_data)))
        axes[1].set_xticklabels([
            f'{int(d)}{ordinal_suffixes[min(int(d)-1, 3)]}' 
            for d in down_data['value']
        ])
        
        # Plot score differential data
        score_data = data[data['situation'] == 'score_differential'].copy()
        sns.barplot(data=score_data, x='value', y='injury_rate', ax=axes[2])
        axes[2].set_title('Injury Rate by Score Differential', pad=15)
        axes[2].set_xlabel('Point Differential', labelpad=10)
        axes[2].set_ylabel('Injuries per 1000 Plays', labelpad=10)
        # Rotate and align the tick labels so they look better
        plt.setp(axes[2].xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Plot time remaining data
        time_data = data[data['situation'] == 'time_remaining'].copy()
        sns.barplot(data=time_data, x='value', y='injury_rate', ax=axes[3])
        axes[3].set_title('Injury Rate by Time Remaining', pad=15)
        axes[3].set_xlabel('Time Remaining (minutes)', labelpad=10)
        axes[3].set_ylabel('Injuries per 1000 Plays', labelpad=10)
        
        # Add gridlines and set consistent y-axis limits for all subplots
        max_rate = data['injury_rate'].max()
        min_rate = data['injury_rate'].min()
        y_padding = (max_rate - min_rate) * 0.1
        
        # Format all subplots
        for ax in [axes[0], axes[2], axes[3]]:
            ax.grid(True, alpha=0.3)
            ax.set_ylim(min_rate - y_padding, max_rate + y_padding)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}'))
            ax.margins(x=0.1)
        
        # Special handling for down plot with dual axis
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim(min_rate - y_padding, max_rate + y_padding)
        axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}'))
        axes[1].margins(x=0.1)
        
        # Adjust bottom margin for x-labels
        plt.subplots_adjust(bottom=0.15)
        
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