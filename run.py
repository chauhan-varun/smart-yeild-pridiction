"""
Utility script to run the Smart Agricultural Yield Prediction System with default parameters.
"""
import os
import subprocess
import sys

def main():
    """Run the yield prediction system with default parameters."""
    print("Starting Smart Agricultural Yield Prediction System...")
    
    # Check if the dependencies are installed
    try:
        import dask
        import pandas
        import numpy
        import matplotlib
        import seaborn
        import xgboost
        print("All required packages are installed.")
    except ImportError as e:
        print(f"Missing dependency: {e}")
        install = input("Do you want to install required dependencies? (y/n): ")
        if install.lower() == 'y':
            subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        else:
            print("Please install dependencies manually using: pip install -r requirements.txt")
            return
    
    # Run the main script
    cmd = [sys.executable, "main.py"]
    subprocess.run(cmd)

if __name__ == "__main__":
    main()
