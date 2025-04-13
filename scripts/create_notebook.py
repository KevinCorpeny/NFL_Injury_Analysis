import json
from pathlib import Path

# Define notebook structure
notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# NFL Injury Analysis - Initial Data Exploration\n",
                "\n",
                "This notebook performs initial exploration of the processed NFL play and injury data."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": [
                "# Import required libraries\n",
                "import pandas as pd\n",
                "import numpy as np\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "from pathlib import Path\n",
                "\n",
                "# Set plotting style\n",
                "plt.style.use('ggplot')  # Use a valid matplotlib style\n",
                "sns.set_style('whitegrid')  # Set seaborn style\n",
                "sns.set_palette('husl')\n",
                "\n",
                "# Set display options\n",
                "pd.set_option('display.max_columns', None)\n",
                "pd.set_option('display.max_rows', 100)\n",
                "pd.set_option('display.width', 1000)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Load Processed Data"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": [
                "# Load processed play data\n",
                "play_data_path = Path('data/processed/processed_plays.parquet')\n",
                "if play_data_path.exists():\n",
                "    plays_df = pd.read_parquet(play_data_path)\n",
                "    print(f\"Loaded {len(plays_df)} plays\")\n",
                "    print(\"\\nColumns:\")\n",
                "    print(plays_df.columns.tolist())\n",
                "else:\n",
                "    print(f\"Play data not found at {play_data_path}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Basic Data Exploration"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": [
                "if 'plays_df' in locals():\n",
                "    # Display basic statistics\n",
                "    print(\"Basic Statistics:\")\n",
                "    print(plays_df.describe())\n",
                "    \n",
                "    # Check for missing values\n",
                "    print(\"\\nMissing Values:\")\n",
                "    missing_values = plays_df.isnull().sum()\n",
                "    print(missing_values[missing_values > 0])"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Play Type Analysis"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": [
                "if 'plays_df' in locals() and 'play_type' in plays_df.columns:\n",
                "    # Count plays by type\n",
                "    play_type_counts = plays_df['play_type'].value_counts()\n",
                "    \n",
                "    # Create bar plot\n",
                "    plt.figure(figsize=(12, 6))\n",
                "    play_type_counts.plot(kind='bar')\n",
                "    plt.title('Distribution of Play Types')\n",
                "    plt.xlabel('Play Type')\n",
                "    plt.ylabel('Count')\n",
                "    plt.xticks(rotation=45)\n",
                "    plt.tight_layout()\n",
                "    plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Temporal Analysis"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": [
                "if 'plays_df' in locals() and 'season' in plays_df.columns:\n",
                "    # Analyze plays by season\n",
                "    season_counts = plays_df['season'].value_counts().sort_index()\n",
                "    \n",
                "    # Create line plot\n",
                "    plt.figure(figsize=(10, 6))\n",
                "    season_counts.plot(kind='line', marker='o')\n",
                "    plt.title('Number of Plays by Season')\n",
                "    plt.xlabel('Season')\n",
                "    plt.ylabel('Number of Plays')\n",
                "    plt.grid(True)\n",
                "    plt.tight_layout()\n",
                "    plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Game Situation Analysis"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": [
                "if 'plays_df' in locals() and all(col in plays_df.columns for col in ['down', 'yards_to_go']):\n",
                "    # Analyze plays by down and distance\n",
                "    plt.figure(figsize=(12, 6))\n",
                "    sns.boxplot(x='down', y='yards_to_go', data=plays_df)\n",
                "    plt.title('Distribution of Yards to Go by Down')\n",
                "    plt.xlabel('Down')\n",
                "    plt.ylabel('Yards to Go')\n",
                "    plt.tight_layout()\n",
                "    plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Save Visualizations"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": [
                "# Create reports directory if it doesn't exist\n",
                "reports_dir = Path('reports/figures')\n",
                "reports_dir.mkdir(parents=True, exist_ok=True)\n",
                "\n",
                "# Save the last figure\n",
                "if 'plt' in locals() and len(plt.get_fignums()) > 0:\n",
                "    plt.savefig(reports_dir / 'yards_to_go_by_down.png')\n",
                "    print(f\"Saved visualization to {reports_dir / 'yards_to_go_by_down.png'}\")"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.8.10"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

# Create notebooks directory if it doesn't exist
notebooks_dir = Path('notebooks')
notebooks_dir.mkdir(exist_ok=True)

# Write notebook to file
notebook_path = notebooks_dir / '01_initial_data_exploration.ipynb'
with open(notebook_path, 'w') as f:
    json.dump(notebook, f, indent=1)

print(f"Created notebook at {notebook_path}") 