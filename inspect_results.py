import pandas as pd
import sys
import os

# Add current directory to path
sys.path.append(os.getcwd())

def inspect_results():
    try:
        df = pd.read_excel("Results.xlsx")
        print("Columns:", df.columns.tolist())
        print("\nFirst 5 rows:")
        print(df.head().to_string())
        
        print("\nUnique Datasets:", df['Dataset'].unique() if 'Dataset' in df.columns else "N/A")
        print("Unique Models:", df['Model'].unique() if 'Model' in df.columns else "N/A")
        print("Unique Phases:", df['Phase'].unique() if 'Phase' in df.columns else "N/A")
        
        # If possible, group by Model/Dataset and get median Q-Error
        if 'Q-Error' in df.columns and 'Model' in df.columns:
             print("\nMedian Q-Error by Model:")
             print(df.groupby('Model')['Q-Error'].median())

    except Exception as e:
        print(f"Error reading excel: {e}")
        # Fallback to csv if excel fails (maybe it's meant to be the csv file?)
        try:
             print("\nAttempting to read final_benchmark.csv...")
             df_csv = pd.read_csv("final_benchmark.csv")
             print("CSV Columns:", df_csv.columns.tolist())
             print(df_csv.head().to_string())
        except Exception as e2:
             print(f"Error reading CSV: {e2}")

if __name__ == "__main__":
    inspect_results()
