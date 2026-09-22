import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
from src.preprocessing.text_preprocessor import SafetyTextPreprocessor

def test_safety_text_preprocessor():
    preprocessor = SafetyTextPreprocessor()
    
    # 1. Test Unicode normalization
    assert preprocessor.preprocess("Café\u00A0Worker") == "cafe worker", "Failed unicode normalization"
    
    # 2. Test whitespace and newline cleanup
    assert preprocessor.preprocess("The   pump \n leaked \r heavily") == "the pump leaked heavily", "Failed whitespace cleanup"
    
    # 3. Test abbreviation expansion
    assert preprocessor.preprocess("IP did not apply LOTO during SIMOPS.") == "injured person did not apply lockout tagout during simultaneous operations.", "Failed abbreviation expansion"
    
    # 4. Test preservation of critical stop words
    assert preprocessor.preprocess("No isolation was applied. Do not start.") == "no isolation was applied. do not start.", "Failed to preserve critical stop words"
    
    # 5. Test basic punctuation removal (preserves hyphens and periods)
    assert preprocessor.preprocess("Pump-12 failed! Valve, V-99 is OK.") == "pump-12 failed valve v-99 is ok.", "Failed punctuation cleanup"
    
    # 6. Test Pandas Series transformation
    s = pd.Series(["LOTO missed!", "IP injured\n\n"])
    s_transformed = preprocessor.transform_series(s)
    assert s_transformed.iloc[0] == "lockout tagout missed"
    assert s_transformed.iloc[1] == "injured person injured"
    
    print("All preprocessing tests passed!")
