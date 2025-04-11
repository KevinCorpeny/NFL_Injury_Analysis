from setuptools import setup, find_packages

setup(
    name="nfl_injury_analysis",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "nfl_data_py>=0.3.3",
        "pandas>=1.3.0",
        "numpy>=1.21.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "requests>=2.26.0",
        "python-dotenv>=0.19.0",
        "jupyter>=1.0.0",
        "scikit-learn>=0.24.0",
    ],
    python_requires=">=3.8",
    author="Your Name",
    description="Analysis of NFL injuries and their correlation with play types",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
) 