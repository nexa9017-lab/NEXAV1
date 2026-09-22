"""
Centralized taxonomy for industrial safety concepts.
Contains Life-Saving Rules, hazards, and barrier failures.
"""

# Life-Saving Rules Definitions
LIFE_SAVING_RULES = {
    "Bypassing Safety Controls": {
        "canonical_name": "Bypassing Safety Controls",
        "description": "Obtain authorization before overriding or disabling safety controls.",
        "keywords": ["bypass", "override", "disable", "interlock", "safety system", "bridge"],
        "precursors": ["disabled safety device", "bypassed interlock", "ignored alarm", "unauthorized override"],
        "related_hazards": ["Process safety event", "Overpressure", "Loss of containment", "Equipment destruction"]
    },
    "Confined Space": {
        "canonical_name": "Confined Space",
        "description": "Obtain authorization before entering a confined space.",
        "keywords": ["vessel", "tank", "entry", "atmosphere", "ventilation", "gas test"],
        "precursors": ["restricted access", "hazardous atmosphere", "poor ventilation", "deep excavation"],
        "related_hazards": ["Toxic atmosphere", "Oxygen deficiency", "Engulfment", "Heat exhaustion", "confined atmosphere"]
    },
    "Driving": {
        "canonical_name": "Driving",
        "description": "Follow safe driving rules.",
        "keywords": ["vehicle", "speed", "seatbelt", "driving", "transport", "truck"],
        "precursors": ["high speed", "distracted driving", "poor road conditions", "heavy vehicle interaction"],
        "related_hazards": ["Vehicle collision", "Rollover", "Pedestrian impact", "Loss of control", "vehicle interaction"]
    },
    "Energy Isolation": {
        "canonical_name": "Energy Isolation",
        "description": "Verify isolation and zero energy before work begins.",
        "keywords": ["loto", "lockout", "tagout", "isolation", "valve", "energy", "pressure"],
        "precursors": ["stored energy", "inadequate isolation", "pressurized equipment", "live electrical", "toxic fluid"],
        "related_hazards": ["Pressurized hydrocarbon release", "Electrical shock", "Stored energy discharge", "Steam leak", "electrical energy"]
    },
    "Hot Work": {
        "canonical_name": "Hot Work",
        "description": "Control flammables and ignition sources.",
        "keywords": ["welding", "grinding", "sparks", "fire", "ignition", "cutting"],
        "precursors": ["ignition source", "flammable atmosphere", "sparks", "hydrocarbon presence"],
        "related_hazards": ["Fire", "Explosion", "Severe burns", "Toxic fumes", "fire/explosion"]
    },
    "Line of Fire": {
        "canonical_name": "Line of Fire",
        "description": "Keep yourself and others out of the line of fire.",
        "keywords": ["barricade", "exclusion zone", "tension", "pinch point", "dropped"],
        "precursors": ["tensioned line", "suspended load", "moving machinery", "pinch point", "pressure test"],
        "related_hazards": ["Struck by moving object", "High pressure release", "Whip from parted cable", "Caught between objects", "line-of-fire exposure"]
    },
    "Safe Mechanical Lifting": {
        "canonical_name": "Safe Mechanical Lifting",
        "description": "Plan lifting operations and control the area.",
        "keywords": ["crane", "lift", "hoist", "rigging", "sling", "load"],
        "precursors": ["suspended load", "heavy lifting", "blind lift", "dynamic loading"],
        "related_hazards": ["Dropped suspended load", "Crane tip-over", "Swinging load", "Rigging failure", "dropped object"]
    },
    "Work at Height": {
        "canonical_name": "Work at Height",
        "description": "Protect yourself against a fall when working at height.",
        "keywords": ["harness", "scaffold", "ladder", "fall", "edge", "lanyard"],
        "precursors": ["working aloft", "unsecured tools", "missing fall protection", "open hole", "edge proximity"],
        "related_hazards": ["Fall from height", "Dropped object", "Unstable platform", "Open grating"]
    },
    "Work Authorization": {
        "canonical_name": "Work Authorization",
        "description": "Work with a valid permit when required.",
        "keywords": ["ptw", "permit", "jsa", "authorization", "simops", "toolbox talk"],
        "precursors": ["unauthorized work", "poor communication", "simops", "inadequate planning"],
        "related_hazards": ["Unidentified hazards", "Simultaneous operations clash", "Uncontrolled work"]
    }
}

# Hazard Taxonomy
HAZARDS = [
    "stored energy",
    "pressurized equipment",
    "high pressure release",
    "hydrocarbon release",
    "fire/explosion",
    "flammable atmosphere",
    "toxic gas",
    "H2S exposure",
    "electrical energy",
    "suspended load",
    "dropped object",
    "line-of-fire exposure",
    "vehicle interaction",
    "moving machinery",
    "confined atmosphere",
    "chemical exposure",
    "fall from height"
]

# Normalized Barrier Failures and their common mappings
BARRIER_FAILURES = {
    "ENERGY_ISOLATION_FAILURE": [
        "loto missing",
        "lock-out not performed",
        "isolation was not verified",
        "loto not applied",
        "isolation valve leaking",
        "bleed valve closed",
        "padlock missing",
        "energy not verified zero"
    ],
    "WORK_AUTHORIZATION_FAILURE": [
        "ptw not issued",
        "jsa not discussed",
        "wrong permit type",
        "expired permit",
        "permit missing",
        "unauthorized"
    ],
    "GAS_TEST_FAILURE": [
        "gas test not performed",
        "gas test expired",
        "no gas check"
    ],
    "FALL_PROTECTION_FAILURE": [
        "harness not clipped",
        "scaffold tag missing",
        "lanyard defective",
        "missing toe boards",
        "not tied off"
    ],
    "LIFTING_CONTROL_FAILURE": [
        "overloaded crane",
        "damaged slings used",
        "outriggers not extended",
        "no tag lines used",
        "lift plan not followed"
    ],
    "LINE_OF_FIRE_CONTROL_FAILURE": [
        "standing in pinch point",
        "exclusion zone breached",
        "tension line snapped",
        "in the line of fire"
    ],
    "TRAFFIC_CONTROL_FAILURE": [
        "speeding",
        "seatbelt not worn",
        "using mobile phone",
        "fatigued driver",
        "brakes defective"
    ],
    "SAFETY_DEVICE_BYPASS": [
        "interlock bridged",
        "alarm disabled",
        "safety valve isolated",
        "override left active"
    ],
    "BARRICADING_FAILURE": [
        "no barricade below",
        "barricades ignored"
    ],
    "COMMUNICATION_FAILURE": [
        "attendant absent",
        "fire watcher absent"
    ],
    "PPE_FAILURE": [
        "missing ppe",
        "safety glasses not worn"
    ],
    "PROCEDURE_NON_COMPLIANCE": [
        "procedure not followed"
    ]
}

def get_normalized_barrier_failure(phrase: str) -> str:
    phrase = phrase.lower().strip()
    for category, mapping_phrases in BARRIER_FAILURES.items():
        if any(p in phrase for p in mapping_phrases):
            return category
    return "UNKNOWN_FAILURE"
