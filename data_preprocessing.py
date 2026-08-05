import pandas as pd
import numpy as np
import os
import glob

def find_dataset():
    """Automatically finds the CSV file in the data directory."""
    print("Searching for dataset in the 'data/' folder...")
    csv_files = glob.glob(os.path.join("data", "*.csv"))
    
    if not csv_files:
        raise FileNotFoundError("❌ No CSV file found! Make sure you extracted the Kaggle file into the 'data/' folder.")
    
    file_path = csv_files[0]
    print(f"✅ Found dataset: {file_path}")
    return file_path

def load_and_clean_data(file_path):
    """Loads the CSV and cleans infinite/missing values."""
    print("\nLoading data into memory (this might take a moment)...")
    df = pd.read_csv(file_path)
    
    # 1. Clean Column Names
    # Network datasets often have trailing spaces in column headers (e.g., ' Flow Duration ')
    df.columns = df.columns.str.strip() 
    print(f"Initial Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")

    # 2. Handle Corrupted Network Data
    print("\nScrubbing infinite and missing values...")
    # Replace Infinity with NaN, then drop all rows with NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    print(f"Cleaned Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")

    # 3. Analyze the Target Variable
    # The ISCX dataset target column is usually named 'Label' or 'Class'
    target_col = 'Label' if 'Label' in df.columns else df.columns[-1]
    
    print("\n--- Network Traffic Class Distribution ---")
    print(df[target_col].value_counts())
    print("------------------------------------------\n")

    return df, target_col

if __name__ == "__main__":
    try:
        # Step 1: Find the file
        dataset_path = find_dataset()
        
        # Step 2: Load and clean
        clean_df, target = load_and_clean_data(dataset_path)
        
        print("🚀 Data preprocessing complete! Ready for Feature Selection.")
        
    except Exception as e:
        print(f"\nError: {e}")