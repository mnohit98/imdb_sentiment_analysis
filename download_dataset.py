#!/usr/bin/env python3
"""
Script to download IMDb dataset from Google Sheets
This script helps download the dataset from the provided Google Sheets link.
"""

import pandas as pd
import os
import sys

def download_dataset():
    """
    Download dataset from Google Sheets
    Note: This requires the dataset to be publicly accessible or you need to use gspread
    """
    print("="*60)
    print("IMDb Dataset Download Helper")
    print("="*60)
    print("\nTo download the dataset:")
    print("1. Open the Google Sheets link in your browser:")
    print("   https://docs.google.com/spreadsheets/d/106x15uz8ccQ6Wvpc8-sYjXisBN8vewS435I7z3wd4sw/edit?gid=1889101679#gid=1889101679")
    print("\n2. Go to File → Download → Comma Separated Values (.csv)")
    print("\n3. Save the file as 'imdb_data.csv' in the 'data/' directory")
    print("\n4. Alternatively, if the sheet is publicly accessible, you can use:")
    print("   - Export URL: https://docs.google.com/spreadsheets/d/106x15uz8ccQ6Wvpc8-sYjXisBN8vewS435I7z3wd4sw/export?format=csv&gid=1889101679")
    print("\n" + "="*60)
    
    # Try to download directly if possible
    export_url = "https://docs.google.com/spreadsheets/d/106x15uz8ccQ6Wvpc8-sYjXisBN8vewS435I7z3wd4sw/export?format=csv&gid=1889101679"
    
    try:
        print("\nAttempting to download dataset directly...")
        df = pd.read_csv(export_url)
        
        # Create data directory if it doesn't exist
        os.makedirs('data', exist_ok=True)
        
        # Save dataset
        output_path = 'data/imdb_data.csv'
        df.to_csv(output_path, index=False)
        
        print(f"✓ Dataset downloaded successfully!")
        print(f"✓ Saved to: {output_path}")
        print(f"✓ Dataset shape: {df.shape}")
        print(f"✓ Columns: {list(df.columns)}")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Direct download failed: {str(e)}")
        print("\nPlease download the dataset manually using the instructions above.")
        return False

if __name__ == "__main__":
    download_dataset()

