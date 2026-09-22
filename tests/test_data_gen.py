import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
from src.data_gen.generate_data import generate_dataset
from src.config import ALL_REPORTS_FILE, TRAIN_FILE, VAL_FILE, TEST_FILE, DB_FILE
import sqlite3

def test_data_generation(tmp_path):
    output_dir = tmp_path / "data"
    df = generate_dataset(output_dir=output_dir)
    
    # 1. Check length
    assert len(df) == 2500, "Should generate 2500 reports"
    
    # 2. Check required columns
    required_cols = [
        "report_id", "date", "site", "location", "report_type", "activity", 
        "description", "sif_label", "sif_potential", "primary_lsr", "lsr_tags",
        "hazard", "barrier_failure", "equipment", "potential_consequence", 
        "actual_consequence", "precursor_tags", "severity"
    ]
    for col in required_cols:
        assert col in df.columns, f"Missing column: {col}"
        
    # 3. Check SIF label constraint
    sif_ratio = df['sif_label'].mean()
    assert 0.20 <= sif_ratio <= 0.35, f"SIF ratio {sif_ratio} out of bounds"
    
    # 4. Check file creation
    assert (output_dir / "all_reports.csv").exists()
    assert (output_dir / "train.csv").exists()
    assert (output_dir / "validation.csv").exists()
    assert (output_dir / "test.csv").exists()
    assert (output_dir / "sif_database.sqlite").exists()
    
    # 5. Check SQLite validity
    conn = sqlite3.connect(output_dir / "sif_database.sqlite")
    db_df = pd.read_sql_query("SELECT * FROM safety_reports", conn)
    assert len(db_df) == 2500
    conn.close()
