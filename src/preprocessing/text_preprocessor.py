import re
import unicodedata

class SafetyTextPreprocessor:
    """
    A reusable preprocessing and feature-engineering layer for safety report text.
    Preserves industrial safety meaning and critical stop words (e.g., 'no', 'not', 'without').
    """
    
    def __init__(self):
        # Common safety and O&G abbreviations mapping
        self.abbreviations = {
            r'\bloto\b': 'lockout tagout',
            r'\bptw\b': 'permit to work',
            r'\bjsa\b': 'job safety analysis',
            r'\bsimops\b': 'simultaneous operations',
            r'\bip\b': 'injured person',
            r'\bppe\b': 'personal protective equipment',
            r'\bhsse\b': 'health safety security environment',
            r'\btoolbox talk\b': 'tbt',
            r'\bh2s\b': 'hydrogen sulfide',
            r'\buoc\b': 'unsafe condition',
            r'\buoa\b': 'unsafe act',
            r'\bnm\b': 'near miss',
            r'\bpip\b': 'process incident',
            r'\bpof\b': 'probability of failure'
        }
    
    def normalize_text(self, text: str) -> str:
        """
        Applies standard normalizations to a single string of text.
        """
        if not isinstance(text, str):
            return ""
            
        # 1. Unicode normalization (handles smart quotes, weird accents, etc.)
        text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        
        # 2. Case normalization
        text = text.lower()
        
        # 3. Newline cleanup
        text = text.replace('\n', ' ').replace('\r', ' ')
        
        # 4. Abbreviation normalization
        for pattern, replacement in self.abbreviations.items():
            text = re.sub(pattern, replacement, text)
            
        # 5. Basic punctuation cleanup (keeping important ones like hyphens in equipment tags might be needed, 
        # but the prompt asked for "basic punctuation cleanup where appropriate". We will remove 
        # non-alphanumeric except for space, hyphen and period to keep sentence structure and equipment IDs).
        # We preserve words like "no", "not", "without" by NOT doing aggressive stopword removal.
        text = re.sub(r'[^\w\s\.-]', ' ', text)
        
        # 6. Repeated-space removal and strip
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def preprocess(self, text: str) -> str:
        """
        Main pipeline to preprocess a single text.
        Currently it wraps normalize_text, but can be extended with tokenization or stemming if needed later.
        For now, it strictly normalizes without removing safety context.
        """
        return self.normalize_text(text)

    def transform_series(self, series):
        """
        Applies the preprocessing pipeline to a pandas Series.
        """
        return series.apply(self.preprocess)
