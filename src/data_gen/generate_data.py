import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
import numpy as np
import random
import uuid
import sqlite3
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split

from src.config import (
    ALL_REPORTS_FILE, TRAIN_FILE, VAL_FILE, TEST_FILE, DB_FILE,
    NUM_REPORTS, RANDOM_SEED, SIF_PROBABILITY
)
from src.data_gen.templates import SITES, LOCATIONS, REPORT_TYPES, LSR_DATA

def random_date(start_year=2020, end_year=2023):
    start = datetime(start_year, 1, 1)
    end = datetime(end_year, 12, 31)
    delta = end - start
    random_days = random.randrange(delta.days)
    return start + timedelta(days=random_days)

def generate_report(is_sif=False):
    lsr = random.choice(list(LSR_DATA.keys()))
    lsr_info = LSR_DATA[lsr]
    
    activity = random.choice(lsr_info["activities"])
    equipment = random.choice(lsr_info["equipment"])
    hazard = random.choice(lsr_info["hazards"])
    barrier_failure = random.choice(lsr_info["barrier_failures"])
    potential_consequence = random.choice(lsr_info["potential_consequences"])
    
    # 10% chance of a multi-label LSR
    lsr_tags = [lsr]
    if random.random() < 0.1:
        extra_lsr = random.choice([k for k in LSR_DATA.keys() if k != lsr])
        lsr_tags.append(extra_lsr)
        
    precursor_tags = random.sample(lsr_info["precursors"], k=min(2, len(lsr_info["precursors"])))
    
    if is_sif:
        template = random.choice(lsr_info["sif_templates"])
        description = template.format(
            activity=activity, 
            equipment=equipment, 
            hazard=hazard, 
            barrier_failure=barrier_failure
        )
        sif_label = 1
        sif_potential = round(random.uniform(0.70, 0.99), 2)
        severity = random.choice(["Medium", "High"])
        actual_consequence = random.choice(["None", "First Aid", "Property Damage"])
    else:
        template = random.choice(lsr_info["non_sif_templates"])
        description = template.format(
            activity=activity, 
            equipment=equipment, 
            hazard=hazard, 
            barrier_failure=barrier_failure
        )
        sif_label = 0
        sif_potential = round(random.uniform(0.01, 0.40), 2)
        severity = "Low"
        actual_consequence = "None"
        # Since it's not SIF, it's a minor thing, clear out major hazards/failures
        hazard = "None"
        barrier_failure = "None"
        potential_consequence = "Minor injury"
        precursor_tags = []

    report = {
        "report_id": f"REP-{uuid.uuid4().hex[:8].upper()}",
        "date": random_date().strftime("%Y-%m-%d"),
        "site": random.choice(SITES),
        "location": random.choice(LOCATIONS),
        "report_type": random.choice(REPORT_TYPES),
        "activity": activity,
        "description": description,
        "sif_label": sif_label,
        "sif_potential": sif_potential,
        "primary_lsr": lsr,
        "lsr_tags": "|".join(lsr_tags),
        "hazard": hazard,
        "barrier_failure": barrier_failure,
        "equipment": equipment,
        "potential_consequence": potential_consequence,
        "actual_consequence": actual_consequence,
        "precursor_tags": "|".join(precursor_tags),
        "severity": severity
    }
    return report

def generate_dataset(output_dir=None):
    if output_dir is None:
        all_file = ALL_REPORTS_FILE
        train_file = TRAIN_FILE
        val_file = VAL_FILE
        test_file = TEST_FILE
        db_file = DB_FILE
    else:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        all_file = out_path / "all_reports.csv"
        train_file = out_path / "train.csv"
        val_file = out_path / "validation.csv"
        test_file = out_path / "test.csv"
        db_file = out_path / "sif_database.sqlite"

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    
    reports = []
    for _ in range(NUM_REPORTS):
        is_sif = random.random() < SIF_PROBABILITY
        reports.append(generate_report(is_sif))
        
    df = pd.DataFrame(reports)
    
    # Save full dataset
    df.to_csv(all_file, index=False)
    
    # Save to SQLite
    conn = sqlite3.connect(db_file)
    df.to_sql('safety_reports', conn, if_exists='replace', index=False)
    conn.close()
    
    # Stratified Split
    train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df['sif_label'], random_state=RANDOM_SEED)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['sif_label'], random_state=RANDOM_SEED)
    
    train_df.to_csv(train_file, index=False)
    val_df.to_csv(val_file, index=False)
    test_df.to_csv(test_file, index=False)
    
    print(f"Generated {len(df)} total reports.")
    print(f"SIF Distribution: {df['sif_label'].mean():.1%}")
    print(f"Train size: {len(train_df)}, Val size: {len(val_df)}, Test size: {len(test_df)}")
    print(f"Data saved to: {all_file.parent}")
    
    return df

if __name__ == "__main__":
    generate_dataset()
