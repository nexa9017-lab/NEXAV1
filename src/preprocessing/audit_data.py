import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
from src.config import TRAIN_FILE, VAL_FILE, TEST_FILE

def audit_dataset(file_path):
    print(f"--- Auditing {file_path.name} ---")
    if not file_path.exists():
        raise FileNotFoundError(f"Missing file: {file_path}")
        
    df = pd.read_csv(file_path)
    
    # Check row counts
    print(f"Row count: {len(df)}")
    
    # Check required columns
    required_cols = [
        "report_id", "date", "site", "location", "report_type", "activity", 
        "description", "sif_label", "sif_potential", "primary_lsr", "lsr_tags",
        "hazard", "barrier_failure", "equipment", "potential_consequence", 
        "actual_consequence", "precursor_tags", "severity"
    ]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
        
    # Check missing descriptions
    missing_desc = df['description'].isna().sum()
    print(f"Missing descriptions: {missing_desc}")
    if missing_desc > 0:
        raise ValueError("Dataset contains missing descriptions!")
        
    # Check duplicate report_ids
    dups = df.duplicated(subset=['report_id']).sum()
    print(f"Duplicate report IDs: {dups}")
    if dups > 0:
        raise ValueError("Dataset contains duplicate report IDs!")
        
    # Class balance for sif_label
    sif_balance = df['sif_label'].value_counts(normalize=True).to_dict()
    print(f"SIF Label Balance: {sif_balance}")
    
    # Report type distribution
    print("Report Type Distribution:")
    print(df['report_type'].value_counts(normalize=True))
    
    # Life-Saving Rule distribution
    print("Primary LSR Distribution:")
    print(df['primary_lsr'].value_counts(normalize=True))
    
    # Activity distribution
    print("Activity Distribution:")
    print(df['activity'].value_counts(normalize=True))
    print("\n")

def main():
    try:
        audit_dataset(TRAIN_FILE)
        audit_dataset(VAL_FILE)
        audit_dataset(TEST_FILE)
        print("All datasets passed the audit successfully!")
    except Exception as e:
        print(f"Audit failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
